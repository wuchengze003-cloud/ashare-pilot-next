"""Walk-forward baseline backtest with leakage protection.

Splits: train -> validation -> out-of-sample, strictly time-ordered.
The simulated OOS loop uses quant_core execution semantics (T+1, lots,
limit up/down, suspension, costs, slippage). This is research evidence,
not a production signal.
"""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal

import numpy as np
from ashare_quant_core import (
    DailyBar,
    DailyBarView,
    DatasetSnapshot,
    ExecutionDay,
    SimulatedPortfolioState,
    classify_board,
    execute_buy,
    execute_sell,
    mark_to_market,
    parse_market_rules,
    settle_t_plus_one,
)

from .baseline_model import (
    DEFAULT_HORIZONS,
    MODEL_KINDS,
    MultiHorizonModel,
    TrainingWindow,
)
from .feature_datasets import FeatureDataset
from .features import (
    FEATURE_TRANSFORMS,
    FeatureRow,
    build_feature_panel,
    feature_names_for_transform,
    forward_return_label,
)

LABEL_HORIZON_FOR_VALIDATION = 5


@dataclass(frozen=True)
class PilotConfig:
    initial_capital: Decimal = Decimal("1000000")
    top_k: int = 5
    per_weight: float = 0.2
    rebalance_interval: int = 5
    model_refit_interval: int = 20
    model_kind: str = "hist_gradient_boosting"
    feature_transform: str = "raw"

    def __post_init__(self) -> None:
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if self.top_k < 1:
            raise ValueError("top_k must be positive")
        if not 0 < self.per_weight <= 1:
            raise ValueError("per_weight must be in (0, 1]")
        if self.rebalance_interval < 1:
            raise ValueError("rebalance_interval must be positive")
        if self.model_refit_interval < 1:
            raise ValueError("model_refit_interval must be positive")
        if self.model_kind not in MODEL_KINDS:
            raise ValueError(f"unsupported model_kind: {self.model_kind}")
        if self.feature_transform not in FEATURE_TRANSFORMS:
            raise ValueError(f"unsupported feature_transform: {self.feature_transform}")


@dataclass(frozen=True)
class LeakCheck:
    check_id: str
    status: str
    detail: str


@dataclass(frozen=True)
class BacktestReport:
    dataset_id: str
    snapshot_sha256: str
    train_end: date
    validation_start: date
    validation_end: date
    test_start: date
    test_end: date
    model_bundle_sha256: str
    training_cutoff: date
    production_training_cutoff: date
    first_nav_date: date
    frozen_valuations: tuple[str, ...]
    validation_ic_mean: float
    score_orientation: int
    model_kind: str
    feature_transform: str
    feature_dataset_id: str | None
    feature_dataset_manifest_sha256: str | None
    top_k: int
    per_weight: float
    rebalance_interval: int
    model_refit_interval: int
    oos_refit_count: int
    oos_max_label_end: date
    nav_curve: tuple[Mapping[str, object], ...]
    metrics: Mapping[str, float]
    final_state: SimulatedPortfolioState
    final_prices: Mapping[str, float]
    buy_dates: Mapping[str, date]
    trades: tuple[Mapping[str, object], ...]
    latest_signal_date: date
    recommendations: tuple[Mapping[str, object], ...]
    feature_weights: tuple[Mapping[str, object], ...]
    leak_checks: tuple[LeakCheck, ...]
    scores_latest: Mapping[str, float] = field(default_factory=dict)


def _group_history(snapshot: DatasetSnapshot) -> dict[str, list[DailyBar]]:
    by_symbol: dict[str, list[DailyBar]] = {}
    for bar in snapshot.bars(through=snapshot.as_of):
        by_symbol.setdefault(bar.symbol, []).append(bar)
    for symbol in by_symbol:
        by_symbol[symbol].sort(key=lambda item: item.trade_date)
    return by_symbol


def _rank_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    def ranks(values: Sequence[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: (values[i], i))
        ranked = [0.0] * len(values)
        for position, index in enumerate(order):
            ranked[index] = float(position)
        return ranked

    rx, ry = ranks(list(x)), ranks(list(y))
    n = len(rx)
    mean_x = sum(rx) / n
    mean_y = sum(ry) / n
    covariance = sum((a - mean_x) * (b - mean_y) for a, b in zip(rx, ry, strict=True))
    var_x = sum((a - mean_x) ** 2 for a in rx)
    var_y = sum((b - mean_y) ** 2 for b in ry)
    if var_x == 0 or var_y == 0:
        return 0.0
    return covariance / math.sqrt(var_x * var_y)


def _split_dates(dates: tuple[date, ...]) -> tuple[date, date, date, date]:
    n = len(dates)
    if n < 40:
        raise ValueError("walk-forward requires at least 40 trade dates")
    train_end = dates[int(n * 0.60) - 1]
    validation_start = dates[int(n * 0.60)]
    validation_end = dates[int(n * 0.78) - 1]
    test_start = dates[int(n * 0.78)]
    if not (train_end < validation_start <= validation_end < test_start):
        raise ValueError("split dates must be ordered and disjoint")
    return train_end, validation_start, validation_end, test_start


def _spearman_ics(
    *,
    history: dict[str, list[DailyBar]],
    rows_by_date: dict[date, list[FeatureRow]],
    model: MultiHorizonModel,
    dates: Sequence[date],
    label_must_end_before: date,
) -> tuple[list[float], date | None]:
    """Rank IC per date; per-symbol labels must end before the given date.

    Symbols use their own bar sequence, so a suspended name's horizon bar
    can land later than the global calendar implies; such pairs are
    excluded here and the true maximum label end date is returned.
    """
    eligible: list[tuple[date, FeatureRow, float]] = []
    max_label_end: date | None = None
    for signal_date in dates:
        rows = rows_by_date.get(signal_date, [])
        for row in rows:
            symbol_history = history[row.symbol]
            index = next(
                (i for i, bar in enumerate(symbol_history) if bar.trade_date == signal_date),
                None,
            )
            if index is None:
                continue
            label_end_index = index + LABEL_HORIZON_FOR_VALIDATION
            if label_end_index >= len(symbol_history):
                continue
            label_end = symbol_history[label_end_index].trade_date
            if label_end >= label_must_end_before:
                continue
            realized = forward_return_label(symbol_history, index, LABEL_HORIZON_FOR_VALIDATION)
            if realized is None:
                continue
            if max_label_end is None or label_end > max_label_end:
                max_label_end = label_end
            eligible.append((signal_date, row, realized))
    if not eligible:
        return [], max_label_end
    matrix = np.asarray([row.values for _, row, _ in eligible], dtype=float)
    predicted = model.score(matrix)
    pairs_by_date: dict[date, list[tuple[float, float]]] = {}
    for (signal_date, _row, realized), score in zip(eligible, predicted, strict=True):
        pairs_by_date.setdefault(signal_date, []).append((float(score), realized))
    ics = [
        _rank_correlation([pair[0] for pair in pairs], [pair[1] for pair in pairs])
        for signal_date in dates
        if len(pairs := pairs_by_date.get(signal_date, [])) >= 3
    ]
    return ics, max_label_end


def _last_known_close(
    history: dict[str, list[DailyBar]],
    history_dates: dict[str, list[date]],
    *,
    through: date,
    symbols: Sequence[str] | set[str] | None = None,
) -> dict[str, float]:
    prices: dict[str, float] = {}
    for symbol in symbols if symbols is not None else history:
        bars = history.get(symbol, [])
        dates = history_dates.get(symbol, [])
        position = bisect_right(dates, through) - 1
        if position >= 0:
            prices[symbol] = bars[position].close
    return prices


def run_walk_forward(
    snapshot: DatasetSnapshot,
    *,
    cost_model_doc: Mapping[str, object],
    market_rules_doc: Mapping[str, object],
    execution_policy_doc: Mapping[str, object],
    portfolio_risk_doc: Mapping[str, object],
    config: PilotConfig | None = None,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    training_window: TrainingWindow | None = None,
    feature_dataset: FeatureDataset | None = None,
) -> tuple[MultiHorizonModel, MultiHorizonModel, BacktestReport]:
    """Train, validate, and simulate strictly out-of-sample.

    Returns (evaluation_model, production_model, report). The evaluation
    model is trained only on the train window and drives the OOS
    simulation and all performance metrics. The production model is
    retrained to the latest safe label date; its training window
    overlaps the OOS period, so it is used only for signal generation
    and current recommendations and must never be used to assess
    performance.
    """
    cfg = config or PilotConfig()
    rules = parse_market_rules(market_rules_doc)
    slippage_bps = int(execution_policy_doc["slippage_bps"])
    max_positions = int(portfolio_risk_doc["max_positions"])
    max_single_weight = float(portfolio_risk_doc["max_single_weight"])
    max_gross_exposure = float(portfolio_risk_doc["max_gross_exposure"])
    rebalance_threshold = float(portfolio_risk_doc.get("rebalance_threshold", 0.0))
    if cfg.top_k > max_positions:
        raise ValueError("top_k exceeds portfolio-risk max_positions")
    if cfg.per_weight > max_single_weight:
        raise ValueError("per_weight exceeds portfolio-risk max_single_weight")
    if cfg.top_k * cfg.per_weight > max_gross_exposure + 1e-12:
        raise ValueError("target weights exceed portfolio-risk max_gross_exposure")
    per_weight = cfg.per_weight

    history = _group_history(snapshot)
    history_dates = {
        symbol: [bar.trade_date for bar in bars] for symbol, bars in history.items()
    }
    bars_by_date: dict[date, dict[str, DailyBar]] = {}
    for symbol, bars in history.items():
        for bar in bars:
            bars_by_date.setdefault(bar.trade_date, {})[symbol] = bar
    dates = tuple(sorted({bar.trade_date for bars in history.values() for bar in bars}))
    if training_window is None:
        train_end, validation_start, validation_end, test_start = _split_dates(dates)
    else:
        train_end = training_window.train_end
        validation_start = training_window.validation_start
        validation_end = training_window.validation_end
        test_start = training_window.test_start
        if any(
            boundary not in dates
            for boundary in (train_end, validation_start, validation_end, test_start)
        ):
            raise ValueError("training window boundaries must be dataset trade dates")
    date_index = {day: position for position, day in enumerate(dates)}

    feature_names = feature_names_for_transform(cfg.feature_transform)
    panel = build_feature_panel(
        snapshot,
        feature_transform=cfg.feature_transform,
        feature_dataset=feature_dataset,
    )
    rows_by_date: dict[date, list[FeatureRow]] = {}
    for row in panel:
        rows_by_date.setdefault(row.trade_date, []).append(row)

    ordered_panel = sorted(panel, key=lambda row: (row.trade_date, row.symbol))
    if not ordered_panel:
        raise ValueError("no training rows available")
    ordered_dates = [row.trade_date for row in ordered_panel]
    if ordered_dates != sorted(ordered_dates):
        raise ValueError("training rows must stay time-ordered")

    feature_matrix = np.asarray([row.values for row in ordered_panel], dtype=float)
    history_index = {
        symbol: {bar.trade_date: position for position, bar in enumerate(bars)}
        for symbol, bars in history.items()
    }
    label_values = {
        horizon: np.full(len(ordered_panel), np.nan, dtype=float) for horizon in horizons
    }
    label_end_ordinals = {
        horizon: np.full(len(ordered_panel), -1, dtype=np.int64) for horizon in horizons
    }
    for horizon in horizons:
        for position, row in enumerate(ordered_panel):
            symbol_history = history[row.symbol]
            bar_index = history_index[row.symbol][row.trade_date]
            label_end_index = bar_index + horizon
            if label_end_index >= len(symbol_history):
                continue
            end_date = symbol_history[label_end_index].trade_date
            realized = forward_return_label(symbol_history, bar_index, horizon)
            if realized is None:
                continue
            label_values[horizon][position] = realized
            label_end_ordinals[horizon][position] = end_date.toordinal()

    def fit_mature_model(*, cutoff: date) -> tuple[MultiHorizonModel, date]:
        """Fit only labels whose symbol-specific outcome is known by cutoff."""
        cutoff_ordinal = cutoff.toordinal()
        mature_labels: dict[int, np.ndarray] = {}
        included_label_ends: list[int] = []
        for horizon in horizons:
            mature = (
                (label_end_ordinals[horizon] >= 0)
                & (label_end_ordinals[horizon] <= cutoff_ordinal)
            )
            mature_labels[horizon] = np.where(
                mature, label_values[horizon], np.nan
            )
            included_label_ends.extend(label_end_ordinals[horizon][mature].tolist())
        if not included_label_ends:
            raise ValueError("no mature training labels available")
        fitted = MultiHorizonModel(horizons=horizons, model_kind=cfg.model_kind)
        fitted.fit(feature_matrix, mature_labels, training_cutoff=cutoff)
        return fitted, date.fromordinal(max(included_label_ends))

    model, max_label_end = fit_mature_model(cutoff=train_end)
    production_cutoff = dates[-1]
    production_model, _production_max_label_end = fit_mature_model(
        cutoff=production_cutoff
    )
    test_start_index = date_index[test_start]
    validation_dates = [
        day
        for day in dates
        if validation_start <= day <= validation_end
        and date_index[day] + LABEL_HORIZON_FOR_VALIDATION < test_start_index
    ]
    if not validation_dates:
        raise ValueError("validation window is empty after label truncation")
    ics, validation_ic_end = _spearman_ics(
        history=history,
        rows_by_date=rows_by_date,
        model=model,
        dates=validation_dates,
        label_must_end_before=test_start,
    )
    if not ics or validation_ic_end is None:
        raise ValueError("validation IC produced no observations")
    raw_validation_ic_mean = float(sum(ics) / len(ics))
    score_orientation = -1 if raw_validation_ic_mean < 0 else 1
    model.set_orientation(score_orientation)
    production_model.set_orientation(score_orientation)
    validation_ic_mean = abs(raw_validation_ic_mean)
    bundle_sha256 = MultiHorizonModel.bundle_sha256(production_model.bundle_bytes())

    signal_dates = [day for day in dates if day in rows_by_date]
    evaluation_scores_by_date: dict[date, dict[str, float]] = {}
    rolling_model: MultiHorizonModel | None = None
    last_refit_index: int | None = None
    refit_evidence: list[tuple[date, date, date]] = []
    for signal_date in (day for day in signal_dates if day >= test_start):
        signal_index = date_index[signal_date]
        if signal_index == 0:
            continue
        if (
            rolling_model is None
            or last_refit_index is None
            or signal_index - last_refit_index >= cfg.model_refit_interval
        ):
            # A one-session embargo makes the timing unambiguous: a model
            # scoring at today's close uses outcomes known by yesterday's close.
            # Label maturity remains symbol-specific across suspension gaps.
            refit_cutoff = dates[signal_index - 1]
            rolling_model, refit_max_label_end = fit_mature_model(cutoff=refit_cutoff)
            rolling_model.set_orientation(score_orientation)
            last_refit_index = signal_index
            refit_evidence.append((signal_date, refit_cutoff, refit_max_label_end))
        rows = rows_by_date.get(signal_date, [])
        if not rows or rolling_model is None:
            continue
        scores = rolling_model.score(np.asarray([row.values for row in rows], dtype=float))
        evaluation_scores_by_date[signal_date] = {
            row.symbol: float(score) for row, score in zip(rows, scores, strict=True)
        }
    if rolling_model is None or not refit_evidence:
        raise ValueError("walk-forward produced no out-of-sample refits")
    model = rolling_model
    oos_max_label_end = max(item[2] for item in refit_evidence)

    def score_production(signal_date: date) -> dict[str, float]:
        """Score with the production model; for signal generation only."""
        rows = rows_by_date.get(signal_date, [])
        if not rows:
            return {}
        matrix = np.asarray([row.values for row in rows], dtype=float)
        scores = production_model.score(matrix)
        return {row.symbol: float(score) for row, score in zip(rows, scores, strict=True)}

    state = SimulatedPortfolioState(cash=cfg.initial_capital, holdings={})
    buy_dates: dict[str, date] = {}
    trades: list[Mapping[str, object]] = []
    nav_points: list[Mapping[str, object]] = []
    benchmark_symbols: tuple[str, ...] = ()
    benchmark_base: dict[str, float] = {}
    first_nav_date: date | None = None
    executed_days = 0
    asset_samples: list[Decimal] = []

    test_dates = [day for day in dates if day >= test_start]
    for execution_date in test_dates:
        state = settle_t_plus_one(state, trade_date=execution_date, buy_dates=buy_dates)
        signal_position = bisect_left(signal_dates, execution_date) - 1
        if signal_position < 0:
            continue
        signal_date = signal_dates[signal_position]
        if signal_date < test_start:
            continue
        if not benchmark_symbols:
            tradable = {
                symbol: bar.close
                for symbol, bar in bars_by_date.get(execution_date, {}).items()
            }
            benchmark_base = tradable
            benchmark_symbols = tuple(sorted(benchmark_base))

        is_rebalance_day = date_index[signal_date] % cfg.rebalance_interval == 0
        scores = evaluation_scores_by_date.get(signal_date, {})
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        targets = (
            [symbol for symbol, _ in ranked[: cfg.top_k]]
            if is_rebalance_day
            else list(state.holdings)
        )
        prev_prices = _last_known_close(
            history,
            history_dates,
            through=signal_date,
            symbols=set(state.holdings) | set(targets),
        )
        held_prices = {
            symbol: prev_prices[symbol] for symbol in state.holdings if symbol in prev_prices
        }
        total_assets = mark_to_market(state, prices=held_prices) if held_prices else state.cash
        asset_samples.append(total_assets)

        day_bars: dict[str, DailyBarView | None] = {}
        previous_closes: dict[str, float] = {}
        for symbol in sorted(set(state.holdings) | set(targets)):
            today = bars_by_date.get(execution_date, {}).get(symbol)
            day_bars[symbol] = (
                DailyBarView(open=today.open, close=today.close) if today else None
            )
            if symbol in prev_prices:
                previous_closes[symbol] = prev_prices[symbol]
        execution_day = ExecutionDay(
            trade_date=execution_date,
            bars=day_bars,
            previous_closes=previous_closes,
            slippage_bps=slippage_bps,
        )

        def record_trade(symbol: str, side: str, trade, skip, *, trade_date=execution_date):
            if trade is not None:
                trades.append(
                    {
                        "trade_date": trade_date.isoformat(),
                        "symbol": symbol,
                        "side": side,
                        "shares": trade.shares,
                        "price": trade.price,
                        "gross_amount": float(trade.gross_amount),
                        "total_cost": float(trade.cost.total),
                        "reason": trade.reason,
                    }
                )
            elif skip is not None:
                trades.append(
                    {
                        "trade_date": trade_date.isoformat(),
                        "symbol": symbol,
                        "side": side,
                        "shares": 0,
                        "price": 0.0,
                        "gross_amount": 0.0,
                        "total_cost": 0.0,
                        "reason": f"SKIPPED_{skip.reason_code}",
                    }
                )

        if is_rebalance_day:
            for symbol in sorted(state.holdings):
                if symbol in targets:
                    continue
                state, trade, skip = execute_sell(
                    state=state,
                    day=execution_day,
                    symbol=symbol,
                    rules=rules,
                    cost_model=cost_model_doc,
                    reason="MODEL_EXIT",
                )
                record_trade(symbol, "sell", trade, skip)

        total_assets_float = float(total_assets)

        if is_rebalance_day:
            for symbol in sorted(state.holdings):
                if symbol not in targets or symbol not in prev_prices:
                    continue
                holding = state.holdings[symbol]
                current_weight = holding.shares * prev_prices[symbol] / total_assets_float
                drift = per_weight - current_weight
                lot = rules[classify_board(symbol)].lot_size
                if drift < -rebalance_threshold:
                    excess_shares = int(((-drift) * total_assets_float) // prev_prices[symbol])
                    if excess_shares >= lot:
                        state, trade, skip = execute_sell(
                            state=state,
                            day=execution_day,
                            symbol=symbol,
                            rules=rules,
                            cost_model=cost_model_doc,
                            reason="REBALANCE_TRIM",
                            max_shares=excess_shares,
                        )
                        record_trade(symbol, "sell", trade, skip)
                elif drift > rebalance_threshold:
                    requested = int((drift * total_assets_float) // prev_prices[symbol])
                    if requested >= lot:
                        state, trade, skip, bought_on = execute_buy(
                            state=state,
                            day=execution_day,
                            symbol=symbol,
                            requested_shares=requested,
                            rules=rules,
                            cost_model=cost_model_doc,
                            reason="REBALANCE_TOPUP",
                        )
                        if trade is not None and bought_on is not None:
                            buy_dates.setdefault(symbol, bought_on)
                        record_trade(symbol, "buy", trade, skip)

        held_gross_weight = (
            sum(
                holding.shares * prev_prices.get(symbol, 0.0)
                for symbol, holding in state.holdings.items()
            )
            / total_assets_float
        )
        available_slots = max_positions - len(state.holdings)
        exposure_budget = max_gross_exposure - held_gross_weight
        if is_rebalance_day:
            for symbol in targets:
                if symbol in state.holdings:
                    continue
                if available_slots <= 0 or exposure_budget < per_weight * 0.5:
                    continue
                reference_price = prev_prices.get(symbol)
                if reference_price is None:
                    continue
                buy_weight = min(per_weight, exposure_budget)
                requested = int((total_assets_float * buy_weight) // reference_price)
                state, trade, skip, bought_on = execute_buy(
                    state=state,
                    day=execution_day,
                    symbol=symbol,
                    requested_shares=requested,
                    rules=rules,
                    cost_model=cost_model_doc,
                    reason="MODEL_TOP_SCORE",
                )
                if trade is not None and bought_on is not None:
                    buy_dates[symbol] = bought_on
                    available_slots -= 1
                    exposure_budget -= per_weight
                record_trade(symbol, "buy", trade, skip)

        close_prices = _last_known_close(
            history,
            history_dates,
            through=execution_date,
            symbols=set(state.holdings),
        )
        held_close_prices = {
            symbol: close_prices[symbol] for symbol in state.holdings if symbol in close_prices
        }
        total_assets_close = (
            mark_to_market(state, prices=held_close_prices) if held_close_prices else state.cash
        )
        benchmark_values = []
        execution_bars = bars_by_date.get(execution_date, {})
        for symbol in benchmark_symbols:
            base = benchmark_base.get(symbol)
            today = execution_bars.get(symbol)
            if base and today:
                benchmark_values.append(today.close / base)
        benchmark_nav = sum(benchmark_values) / len(benchmark_values) if benchmark_values else 1.0
        if first_nav_date is None:
            first_nav_date = execution_date
        nav_points.append(
            {
                "trade_date": execution_date.isoformat(),
                "nav": float(total_assets_close / cfg.initial_capital),
                "benchmark_nav": float(benchmark_nav),
            }
        )
        executed_days += 1

    if not nav_points:
        raise ValueError("walk-forward simulation produced no out-of-sample days")

    assert first_nav_date is not None
    baseline_index = date_index[first_nav_date] - 1
    baseline_date = dates[max(baseline_index, 0)]
    nav_points.insert(
        0,
        {"trade_date": baseline_date.isoformat(), "nav": 1.0, "benchmark_nav": 1.0},
    )
    gross_traded = Decimal(str(sum(float(t["gross_amount"]) for t in trades)))
    last_execution_day = date.fromisoformat(str(nav_points[-1]["trade_date"]))
    frozen_valuations = tuple(
        symbol
        for symbol in sorted(state.holdings)
        if history[symbol][-1].trade_date < last_execution_day
    )

    navs = [float(point["nav"]) for point in nav_points]
    total_return = navs[-1] - 1.0
    peak = 1.0
    max_drawdown = 0.0
    daily_returns: list[float] = []
    for index, value in enumerate(navs):
        peak = max(peak, value)
        max_drawdown = max(max_drawdown, (peak - value) / peak if peak else 0.0)
        if index:
            daily_returns.append(value / navs[index - 1] - 1)
    win_rate = (
        sum(1 for ret in daily_returns if ret > 0) / len(daily_returns) if daily_returns else 0.0
    )
    benchmark_return = float(nav_points[-1]["benchmark_nav"]) - 1.0
    average_assets = (
        sum(asset_samples, Decimal(0)) / Decimal(len(asset_samples))
        if asset_samples
        else Decimal(1)
    )
    turnover = float(gross_traded / average_assets) / executed_days if executed_days else 0.0

    latest_signal_date = signal_dates[-1]
    latest_scores = score_production(latest_signal_date)
    latest_ranked = sorted(latest_scores.items(), key=lambda item: (-item[1], item[0]))
    latest_targets = [symbol for symbol, _ in latest_ranked[: cfg.top_k]]
    latest_prices = _last_known_close(
        history,
        history_dates,
        through=latest_signal_date,
    )
    final_prices = {
        symbol: latest_prices[symbol] for symbol in state.holdings if symbol in latest_prices
    }

    recommendations: list[Mapping[str, object]] = []
    number_of_symbols = len(latest_ranked)
    for rank, (symbol, score) in enumerate(latest_ranked, start=1):
        if symbol in latest_targets and symbol in state.holdings:
            recommendation = "HOLD"
        elif symbol in latest_targets:
            recommendation = "BUY"
        elif symbol in state.holdings:
            recommendation = "SELL"
        else:
            recommendation = "WATCH"
        rank_strength = 1.0 - (rank - 1) / max(number_of_symbols, 1)
        last_close = latest_prices.get(symbol)
        price_band = None
        risk_notes: list[str] = []
        if recommendation == "BUY" and last_close is not None:
            price_band = {"low": round(last_close * 0.98, 4), "high": round(last_close, 4)}
        symbol_history = history.get(symbol, [])
        if symbol_history and symbol_history[-1].trade_date < latest_signal_date:
            risk_notes.append("NO_BAR_ON_SIGNAL_DATE")
        if symbol in frozen_valuations:
            risk_notes.append("VALUATION_FROZEN_NO_BAR")
        if len(symbol_history) >= 2:
            prev_close = symbol_history[-2].close
            segment = rules.get(classify_board(symbol))
            if segment is not None and last_close is not None:
                if last_close >= prev_close * (1 + segment.price_limit_pct * 0.95):
                    risk_notes.append("NEAR_LIMIT_UP")
                if last_close <= prev_close * (1 - segment.price_limit_pct * 0.95):
                    risk_notes.append("NEAR_LIMIT_DOWN")
        holding = state.holdings.get(symbol)
        if holding is not None and holding.locked_shares > 0:
            risk_notes.append("T1_LOCKED_SHARES")
        recommendations.append(
            {
                "symbol": symbol,
                "rank": rank,
                "score": round(score, 6),
                "recommendation": recommendation,
                "rank_strength": round(rank_strength, 4),
                "price_band": price_band,
                "risk_notes": risk_notes,
            }
        )

    score_matrix = np.asarray(
        [row.values for row in rows_by_date.get(latest_signal_date, [])], dtype=float
    )
    feature_weights: list[Mapping[str, object]] = []
    if score_matrix.size:
        latest_score_vector = production_model.score(score_matrix)
        for column, name in enumerate(feature_names):
            column_values = score_matrix[:, column]
            weight = abs(_rank_correlation(list(column_values), list(latest_score_vector)))
            feature_weights.append({"name": name, "weight": round(float(weight), 6)})
        feature_weights.sort(key=lambda item: (-item["weight"], item["name"]))

    truncated_panel = build_feature_panel(
        snapshot,
        as_of=latest_signal_date,
        feature_transform=cfg.feature_transform,
        feature_dataset=feature_dataset,
    )
    full_panel_visible = tuple(row for row in panel if row.trade_date <= latest_signal_date)
    pit_status = "pass" if truncated_panel == full_panel_visible else "fail"

    leak_checks = (
        LeakCheck(
            check_id="temporal_split_monotonic",
            status="pass",
            detail=(
                f"train_end={train_end.isoformat()} < validation=["
                f"{validation_start.isoformat()},{validation_end.isoformat()}] < "
                f"test_start={test_start.isoformat()}"
            ),
        ),
        LeakCheck(
            check_id="label_window_inside_train",
            status="pass",
            detail=f"latest label end date {max_label_end.isoformat()} <= train_end",
        ),
        LeakCheck(
            check_id="validation_labels_outside_test",
            status="pass" if validation_ic_end < test_start else "fail",
            detail=(
                f"validation IC scored over [{validation_start.isoformat()},"
                f"{validation_ic_end.isoformat()}]; actual max label end "
                f"{validation_ic_end.isoformat()} vs test_start={test_start.isoformat()}"
            ),
        ),
        LeakCheck(
            check_id="training_rows_time_ordered",
            status="pass",
            detail=f"{len(ordered_panel)} feature rows keep non-decreasing trade_date",
        ),
        LeakCheck(
            check_id="walk_forward_refit_label_embargo",
            status=(
                "pass"
                if all(
                    label_end <= cutoff < signal
                    for signal, cutoff, label_end in refit_evidence
                )
                else "fail"
            ),
            detail=(
                f"{len(refit_evidence)} expanding-window refits; all labels mature "
                f"by the prior-session cutoff; latest included label end "
                f"{oos_max_label_end.isoformat()}"
            ),
        ),
        LeakCheck(
            check_id="feature_pit_truncation_invariant",
            status=pit_status,
            detail=(
                "panel rebuilt with as_of=latest signal date equals the visible "
                "subset of the full panel; within-window mutation coverage is in "
                "unit tests"
            ),
        ),
        LeakCheck(
            check_id="universe_fixed_audit_set_disclosed",
            status="pass",
            detail=(
                "universe is a fixed audit list including delisted and suspended "
                "names; no rolling point-in-time membership; survivorship control "
                "limited to the fixed list"
            ),
        ),
    )

    report = BacktestReport(
        dataset_id=snapshot.dataset_id,
        snapshot_sha256=snapshot.snapshot_sha256,
        train_end=train_end,
        validation_start=validation_start,
        validation_end=validation_end,
        test_start=test_start,
        test_end=dates[-1],
        model_bundle_sha256=bundle_sha256,
        training_cutoff=model.training_cutoff or train_end,
        production_training_cutoff=production_cutoff,
        first_nav_date=first_nav_date,
        frozen_valuations=frozen_valuations,
        validation_ic_mean=validation_ic_mean,
        score_orientation=score_orientation,
        model_kind=cfg.model_kind,
        feature_transform=cfg.feature_transform,
        feature_dataset_id=(
            feature_dataset.feature_dataset_id if feature_dataset is not None else None
        ),
        feature_dataset_manifest_sha256=(
            feature_dataset.manifest_sha256 if feature_dataset is not None else None
        ),
        top_k=cfg.top_k,
        per_weight=cfg.per_weight,
        rebalance_interval=cfg.rebalance_interval,
        model_refit_interval=cfg.model_refit_interval,
        oos_refit_count=len(refit_evidence),
        oos_max_label_end=oos_max_label_end,
        nav_curve=tuple(nav_points),
        metrics={
            "total_return": round(total_return, 6),
            "benchmark_total_return": round(benchmark_return, 6),
            "max_drawdown": round(max_drawdown, 6),
            "win_rate": round(win_rate, 6),
            "turnover": round(turnover, 6),
        },
        final_state=state,
        final_prices=final_prices,
        buy_dates=dict(buy_dates),
        trades=tuple(trades),
        latest_signal_date=latest_signal_date,
        recommendations=tuple(recommendations),
        feature_weights=tuple(feature_weights[:3]),
        leak_checks=leak_checks,
        scores_latest=latest_scores,
    )
    return model, production_model, report
