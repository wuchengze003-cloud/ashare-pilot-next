"""Walk-forward baseline backtest with leakage protection.

Splits: train -> validation -> out-of-sample, strictly time-ordered.
The simulated OOS loop uses quant_core execution semantics (T+1, lots,
limit up/down, suspension, costs, slippage). This is research evidence,
not a production signal.
"""

from __future__ import annotations

import math
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

from .baseline_model import DEFAULT_HORIZONS, MultiHorizonModel
from .features import FEATURE_NAMES, FeatureRow, build_feature_panel, forward_return_label

LABEL_HORIZON_FOR_VALIDATION = 5


@dataclass(frozen=True)
class PilotConfig:
    initial_capital: Decimal = Decimal("500000")
    top_k: int = 5
    per_weight: float = 0.2

    def __post_init__(self) -> None:
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if self.top_k < 1:
            raise ValueError("top_k must be positive")
        if not 0 < self.per_weight <= 1:
            raise ValueError("per_weight must be in (0, 1]")


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
    ics: list[float] = []
    max_label_end: date | None = None
    for signal_date in dates:
        rows = rows_by_date.get(signal_date, [])
        pairs: list[tuple[float, float]] = []
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
            score = float(model.score(np.asarray(row.values, dtype=float).reshape(1, -1))[0])
            pairs.append((score, realized))
        if len(pairs) >= 3:
            ics.append(_rank_correlation([p[0] for p in pairs], [p[1] for p in pairs]))
    return ics, max_label_end


def _last_known_close(history: dict[str, list[DailyBar]], *, through: date) -> dict[str, float]:
    prices: dict[str, float] = {}
    for symbol, bars in history.items():
        visible = [bar for bar in bars if bar.trade_date <= through]
        if visible:
            prices[symbol] = visible[-1].close
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
    per_weight = min(cfg.per_weight, max_single_weight)

    history = _group_history(snapshot)
    dates = tuple(sorted({bar.trade_date for bars in history.values() for bar in bars}))
    train_end, validation_start, validation_end, test_start = _split_dates(dates)
    date_index = {day: position for position, day in enumerate(dates)}
    train_end_index = date_index[train_end]

    panel = build_feature_panel(snapshot)
    rows_by_date: dict[date, list[FeatureRow]] = {}
    for row in panel:
        rows_by_date.setdefault(row.trade_date, []).append(row)

    training_rows = sorted(
        (row for row in panel if row.trade_date <= train_end),
        key=lambda row: (row.trade_date, row.symbol),
    )
    if not training_rows:
        raise ValueError("no training rows available")
    ordered_dates = [row.trade_date for row in training_rows]
    if ordered_dates != sorted(ordered_dates):
        raise ValueError("training rows must stay time-ordered")

    feature_matrix = np.asarray([row.values for row in training_rows], dtype=float)
    labels_by_horizon: dict[int, np.ndarray] = {}
    max_label_end = train_end
    for horizon in horizons:
        labels = np.full(len(training_rows), np.nan, dtype=float)
        for position, row in enumerate(training_rows):
            symbol_history = history[row.symbol]
            index = date_index[row.trade_date]
            target_index = index + horizon
            if target_index > train_end_index:
                continue
            bar_index = next(
                (
                    i
                    for i, bar in enumerate(symbol_history)
                    if bar.trade_date == row.trade_date
                ),
                None,
            )
            if bar_index is None:
                continue
            realized = forward_return_label(symbol_history, bar_index, horizon)
            if realized is None:
                continue
            labels[position] = realized
            end_date = symbol_history[bar_index + horizon].trade_date
            if end_date > max_label_end:
                raise ValueError("label window escaped the training cutoff")
        labels_by_horizon[horizon] = labels

    model = MultiHorizonModel(horizons=horizons)
    model.fit(feature_matrix, labels_by_horizon, training_cutoff=train_end)

    max_horizon = max(horizons)
    production_cutoff = dates[len(dates) - 1 - max_horizon]
    production_index = date_index[production_cutoff]
    production_rows = sorted(
        (row for row in panel if row.trade_date <= production_cutoff),
        key=lambda row: (row.trade_date, row.symbol),
    )
    production_matrix = np.asarray([row.values for row in production_rows], dtype=float)
    production_labels: dict[int, np.ndarray] = {}
    for horizon in horizons:
        labels = np.full(len(production_rows), np.nan, dtype=float)
        for position, row in enumerate(production_rows):
            index = date_index[row.trade_date]
            if index + horizon > production_index:
                continue
            symbol_history = history[row.symbol]
            bar_index = next(
                (
                    i
                    for i, bar in enumerate(symbol_history)
                    if bar.trade_date == row.trade_date
                ),
                None,
            )
            if bar_index is None:
                continue
            label_end_index = bar_index + horizon
            if label_end_index >= len(symbol_history):
                continue
            if symbol_history[label_end_index].trade_date > production_cutoff:
                continue
            realized = forward_return_label(symbol_history, bar_index, horizon)
            if realized is not None:
                labels[position] = realized
        production_labels[horizon] = labels
    production_model = MultiHorizonModel(horizons=horizons)
    production_model.fit(
        production_matrix,
        production_labels,
        training_cutoff=production_cutoff,
    )
    bundle_sha256 = MultiHorizonModel.bundle_sha256(production_model.bundle_bytes())

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
    validation_ic_mean = float(sum(ics) / len(ics))

    signal_dates = [day for day in dates if day in rows_by_date]

    def score_date(signal_date: date) -> dict[str, float]:
        rows = rows_by_date.get(signal_date, [])
        if not rows:
            return {}
        matrix = np.asarray([row.values for row in rows], dtype=float)
        scores = model.score(matrix)
        return {row.symbol: float(score) for row, score in zip(rows, scores, strict=True)}

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
        prior_signals = [day for day in signal_dates if day < execution_date and day >= test_start]
        if not prior_signals:
            continue
        signal_date = prior_signals[-1]
        if not benchmark_symbols:
            tradable: dict[str, float] = {}
            for symbol, bars in history.items():
                todays = [bar for bar in bars if bar.trade_date == execution_date]
                if todays:
                    tradable[symbol] = todays[0].close
            benchmark_base = tradable
            benchmark_symbols = tuple(sorted(benchmark_base))

        scores = score_date(signal_date)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        targets = [symbol for symbol, _ in ranked[: cfg.top_k]]
        prev_prices = _last_known_close(history, through=signal_date)
        held_prices = {
            symbol: prev_prices[symbol] for symbol in state.holdings if symbol in prev_prices
        }
        total_assets = (
            mark_to_market(state, prices=held_prices) if held_prices else state.cash
        )
        asset_samples.append(total_assets)

        day_bars: dict[str, DailyBarView | None] = {}
        previous_closes: dict[str, float] = {}
        for symbol in sorted(set(state.holdings) | set(targets)):
            bars = history.get(symbol, [])
            todays = [bar for bar in bars if bar.trade_date == execution_date]
            day_bars[symbol] = (
                DailyBarView(open=todays[0].open, close=todays[0].close) if todays else None
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

        held_gross_weight = sum(
            holding.shares * prev_prices.get(symbol, 0.0)
            for symbol, holding in state.holdings.items()
        ) / total_assets_float
        available_slots = max_positions - len(state.holdings)
        exposure_budget = max_gross_exposure - held_gross_weight
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

        close_prices = _last_known_close(history, through=execution_date)
        held_close_prices = {
            symbol: close_prices[symbol] for symbol in state.holdings if symbol in close_prices
        }
        total_assets_close = (
            mark_to_market(state, prices=held_close_prices) if held_close_prices else state.cash
        )
        benchmark_values = []
        for symbol in benchmark_symbols:
            base = benchmark_base.get(symbol)
            todays = [bar for bar in history[symbol] if bar.trade_date == execution_date]
            if base and todays:
                benchmark_values.append(todays[0].close / base)
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
    turnover = (
        float(gross_traded / average_assets) / executed_days if executed_days else 0.0
    )

    latest_signal_date = signal_dates[-1]
    latest_scores = score_production(latest_signal_date)
    latest_ranked = sorted(latest_scores.items(), key=lambda item: (-item[1], item[0]))
    latest_targets = [symbol for symbol, _ in latest_ranked[: cfg.top_k]]
    latest_prices = _last_known_close(history, through=latest_signal_date)
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
        for column, name in enumerate(FEATURE_NAMES):
            column_values = score_matrix[:, column]
            weight = abs(_rank_correlation(list(column_values), list(latest_score_vector)))
            feature_weights.append({"name": name, "weight": round(float(weight), 6)})
        feature_weights.sort(key=lambda item: (-item["weight"], item["name"]))

    truncated_panel = build_feature_panel(snapshot, as_of=latest_signal_date)
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
            detail=f"{len(training_rows)} training rows keep non-decreasing trade_date",
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
        training_cutoff=train_end,
        production_training_cutoff=production_cutoff,
        first_nav_date=first_nav_date,
        frozen_valuations=frozen_valuations,
        validation_ic_mean=validation_ic_mean,
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
        trades=tuple(trades[-40:]),
        latest_signal_date=latest_signal_date,
        recommendations=tuple(recommendations),
        feature_weights=tuple(feature_weights[:3]),
        leak_checks=leak_checks,
        scores_latest=latest_scores,
    )
    return model, production_model, report
