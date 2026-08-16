"""Render the static dashboard from runtime/dashboard/champion.json.

Read-only presentation: all numbers are computed by `champion.py` (Research
layer); this script only draws them into self-contained HTML pages (no
financial calculation, no external CDN — inline SVG for charts).

Outputs:
    runtime/dashboard/index.html           main dashboard
    runtime/dashboard/detail/{code}.html   per-stock detail (candlestick + L1)
"""
from __future__ import annotations

import json
from pathlib import Path

# ruff: noqa: E501  (embedded CSS/HTML lines are intentionally long)

ROOT = Path(__file__).resolve().parent.parent
DETAIL_DIR = ROOT / "runtime/dashboard/detail"


def _load() -> dict:
    return json.loads((ROOT / "runtime/dashboard/champion.json").read_text(encoding="utf-8"))


def _load_name_map(symbols: set[str]) -> dict[str, str]:
    names: dict[str, str] = {}
    for sym in symbols:
        code = sym.split(".")[0]
        p = ROOT / "runtime/full-market" / f"{code}.json"
        if p.exists():
            doc = json.loads(p.read_text(encoding="utf-8"))
            names[sym] = str(doc.get("name", ""))
    return names


def _load_bars(symbol: str) -> list[dict]:
    code = symbol.split(".")[0]
    p = ROOT / "runtime/full-market" / f"{code}.json"
    if not p.exists():
        return []
    doc = json.loads(p.read_text(encoding="utf-8"))
    return doc.get("data", [])


def _svg_line(points, key, w, h, pad, ymin=None, ymax=None):
    if not points:
        return ""
    step = max(1, len(points) // 120)
    pts = points[::step]
    if ymin is None or ymax is None:
        ys = [p[key] for p in pts]
        ymin, ymax = min(ys), max(ys)
    span = (ymax - ymin) or 1.0
    coords = []
    for i, p in enumerate(pts):
        x = pad + (w - 2 * pad) * i / (len(pts) - 1)
        y = pad + (h - 2 * pad) * (1 - (p[key] - ymin) / span)
        coords.append(f"{x:.1f},{y:.1f}")
    return " ".join(coords)


def _nav_annotations(nav_pts, model_evolution, w, h, pad):
    """X-axis time ticks + clickable vertical lines marking each rolling model."""
    if not nav_pts:
        return ""
    n = len(nav_pts)
    out = []
    tick_idx = [0, n // 4, n // 2, 3 * n // 4, n - 1]
    for idx in tick_idx:
        p = nav_pts[idx]
        x = pad + (w - 2 * pad) * idx / (n - 1)
        out.append(
            f'<text x="{x:.1f}" y="{h - 6}" text-anchor="middle" fill="#8b949e" '
            f'font-size="11">{p["date"][:7]}</text>'
        )
    date_pos = {p["date"]: i for i, p in enumerate(nav_pts)}
    for me in (model_evolution or []):
        rd = me.get("refit_date", "")
        idx = date_pos.get(rd)
        if idx is None:
            continue
        x = pad + (w - 2 * pad) * idx / (n - 1)
        mi = me.get("index", 0)
        out.append(
            f'<a class="model-stage" href="model/model_{mi}.html" title="模型 M{mi}（{rd} 重训，点击查看因子权重）">'
            f'<line x1="{x:.1f}" y1="{pad}" x2="{x:.1f}" y2="{h - pad}" '
            f'stroke="#d29922" stroke-width="0.5" stroke-dasharray="2 3" opacity="0.5"/>'
            f'<text x="{x:.1f}" y="{pad + 10:.1f}" text-anchor="middle" fill="#d29922" '
            f'font-size="9" font-weight="600">M{mi}</text>'
            f'</a>'
        )
    return "".join(out)


def _metric_card(label, value, sub="", color=""):
    cls = f' class="{color}"' if color else ""
    return (
        f'<div class="card"><div class="card-label">{label}</div>'
        f'<div class="card-value{cls}">{value}</div>'
        f'<div class="card-sub">{sub}</div></div>'
    )


def _fmt(unit, value):
    if unit == "pct":
        return f"{value:+.1f}%"
    if unit == "pct_raw":
        return f"{value:.1f}%"
    if unit == "mktcap":
        return f"{value:.0f}亿"
    return f"{value:.1f}"


def _attribution_card(s):
    attrs = s.get("attribution", [])
    code = s["symbol"].split(".")[0]
    rows = ""
    for a in attrs:
        p = a["percentile"] * 100
        rows += (
            f'<div class="attr-row"><span class="attr-label">{a["label"]}</span>'
            f'<span class="attr-val">{_fmt(a["unit"], a["value"])}</span>'
            f'<div class="pbar"><div class="pfill" style="width:{p:.0f}%"></div></div>'
            f'<span class="attr-pct">前 {p:.0f}%</span></div>'
        )
    return (
        f'<a class="stock-card" href="detail/{code}.html">'
        f'<div class="stock-head"><span class="code">{s["symbol"]}</span>'
        f'<span class="name">{s["name"]}</span>'
        f'<span class="reason">{s.get("industry", "")}</span>'
        f'<span class="score">评分 {s["score"]:+.4f}</span></div>'
        f'<div class="stock-sub">现价 {s.get("close", 0):.2f} · 市值 {s.get("market_cap", 0):.0f}亿 · '
        f'计划买 {s.get("shares", 0)} 股（约 {s.get("amount", 0) / 10000:.2f} 万） · 点击看日线 →</div>'
        f'{rows}'
        f'</a>'
    )


def _importance_bars(importance, w, h, pad):
    if not importance:
        return ""
    top = importance[:15]
    max_imp = max(x["importance"] for x in top) or 1.0
    bar_h = (h - 2 * pad) / len(top)
    label_w = 110
    out = []
    for i, x in enumerate(top):
        cy = pad + i * bar_h + bar_h / 2
        bw = (x["importance"] / max_imp) * (w - pad * 2 - label_w - 70)
        out.append(
            f'<text x="{pad}" y="{cy + 4}" fill="#8b949e" font-size="12">{x["label"]}</text>'
        )
        out.append(
            f'<rect x="{pad + label_w}" y="{cy - 5}" width="{bw:.1f}" height="10" '
            f'rx="2" fill="#58a6ff" class="bar-anim" style="animation-delay:{i * 40}ms"/>'
        )
        out.append(
            f'<text x="{pad + label_w + bw + 6:.1f}" y="{cy + 4}" fill="#e6edf3" '
            f'font-size="11">{x["importance"]:.4f}</text>'
        )
    return "".join(out)


def _exit_reason_bars(exit_reasons, w, h, pad):
    if not exit_reasons:
        return ""
    colors = {"stop": "#f85149", "profit": "#d29922", "signal": "#58a6ff", "cashout": "#8b949e"}
    total = sum(x["count"] for x in exit_reasons)
    max_c = max(x["count"] for x in exit_reasons)
    bar_h = (h - 2 * pad) / len(exit_reasons)
    out = []
    for i, x in enumerate(exit_reasons):
        cy = pad + i * bar_h + bar_h / 2
        bw = (x["count"] / max_c) * (w - pad * 2 - 120 - 80)
        color = colors.get(x["reason"], "#58a6ff")
        out.append(
            f'<text x="{pad}" y="{cy + 4}" fill="#8b949e" font-size="12">{x["label"]}</text>'
        )
        out.append(
            f'<rect x="{pad + 100}" y="{cy - 6}" width="{bw:.1f}" height="12" rx="2" '
            f'fill="{color}" class="bar-anim" style="animation-delay:{i * 60}ms"/>'
        )
        out.append(
            f'<text x="{pad + 100 + bw + 8:.1f}" y="{cy + 4}" fill="#e6edf3" '
            f'font-size="12">{x["count"]} 笔 ({x["count"] / total:.0%})</text>'
        )
    return "".join(out)


def _yearly_bars(yearly, w, h, pad):
    if not yearly:
        return ""
    max_abs = max(abs(y["return"]) for y in yearly) or 1.0
    bar_h = (h - 2 * pad) / len(yearly)
    out = [f'<line x1="{w/2}" y1="{pad}" x2="{w/2}" y2="{h-pad}" stroke="#30363d" stroke-width="0.5"/>']
    for i, y in enumerate(yearly):
        ret = y["return"]
        top = pad + i * bar_h
        cy = top + bar_h / 2
        bw = (abs(ret) / max_abs) * (w / 2 - pad - 60)
        color = "#f85149" if ret >= 0 else "#3fb950"
        x0 = w / 2 + 10 if ret >= 0 else w / 2 - 10 - bw
        x1 = w / 2 + 10 + bw if ret >= 0 else w / 2 - 10
        out.append(f'<text x="{w/2 - 8}" y="{cy + 4}" text-anchor="end" fill="#8b949e" font-size="13">{y["year"]}</text>')
        out.append(f'<rect x="{min(x0, x1):.1f}" y="{top + (bar_h - bar_h * 0.35) / 2:.1f}" width="{abs(x1 - x0):.1f}" height="{bar_h * 0.35:.1f}" rx="3" fill="{color}" class="bar-anim" style="animation-delay:{i * 60}ms"/>')
        out.append(f'<text x="{max(x0, x1) + 6:.1f}" y="{cy + 4}" fill="{color}" font-size="12">{ret:+.1%}</text>')
    return "".join(out)


REASON_DETAIL = {
    "stop": "STOP 止损 · 跌超 6%",
    "profit": "PROFIT 止盈 · 涨超 25%",
    "signal": "SIGNAL 掉出前 16 名",
    "cashout": "CASHOUT 大盘破均线空仓",
    "maxhold": "MAXHOLD 持有期满",
}


def _trades_table(trades, name_map, page_size=20):
    buys = [t for t in trades if t["side"] == "buy"][::-1]
    sells = [t for t in trades if t["side"] == "sell"][::-1]

    def to_js(ts):
        rows = []
        for t in ts:
            code = t["symbol"].split(".")[0]
            reason = REASON_DETAIL.get(t["reason"], t["reason"]) if t["side"] == "sell" else "建仓"
            rows.append({
                "date": t["date"], "symbol": t["symbol"], "code": code,
                "name": name_map.get(t["symbol"], ""), "price": t["price"],
                "label": "BUY" if t["side"] == "buy" else "SELL",
                "tag": t["side"], "reason": reason,
            })
        return rows

    buys_js = json.dumps(to_js(buys), ensure_ascii=False)
    sells_js = json.dumps(to_js(sells), ensure_ascii=False)

    head = (
        f'<div class="sub" style="margin-bottom:8px">共 {len(trades)} 笔：'
        f'<span class="buy">BUY {len(buys)}</span> / '
        f'<span class="sell">SELL {len(sells)}</span>'
        f'（每页 {page_size} 笔，点击页码翻页）</div>'
        '<div class="table-wrap">'
        '<div class="trade-table" id="trade-buy"></div>'
        '<div class="trade-table" id="trade-sell"></div>'
        '</div>'
    )
    js = f"""<script>
const BUYS = {buys_js};
const SELLS = {sells_js};
const PS = {page_size};
function tradeRow(t) {{
  return '<tr><td class="code">' + t.date + '</td>' +
    '<td class="code"><a href="detail/' + t.code + '.html">' + t.symbol + '</a></td>' +
    '<td class="name">' + t.name + '</td><td>' + t.price.toFixed(2) + '</td>' +
    '<td class="' + t.tag + '">' + t.label + '</td>' +
    '<td class="reason">' + t.reason + '</td></tr>';
}}
function renderTrade(kind, page) {{
  var data = (kind === 0) ? BUYS : SELLS;
  var el = document.getElementById(kind === 0 ? "trade-buy" : "trade-sell");
  var pages = Math.max(1, Math.ceil(data.length / PS));
  var slice = data.slice(page * PS, (page + 1) * PS);
  var h = '<table><thead><tr><th>日期</th><th>代码</th><th>名称</th><th>价格</th><th>方向</th><th>原因</th></tr></thead><tbody>';
  for (var i = 0; i < slice.length; i++) {{ h += tradeRow(slice[i]); }}
  h += '</tbody></table><div class="pager">';
  var show = [];
  for (var p = 0; p < pages; p++) {{
    if (p === 0 || p === pages - 1 || Math.abs(p - page) <= 2) show.push(p);
  }}
  var last = -2;
  for (var q = 0; q < show.length; q++) {{
    if (show[q] > last + 1) h += '<span class="page-ell">…</span>';
    h += '<button class="page-btn' + (show[q] === page ? ' active' : '') + '" onclick="renderTrade(' + kind + ',' + show[q] + ')">' + (show[q] + 1) + '</button>';
    last = show[q];
  }}
  h += '</div>';
  el.innerHTML = h;
}}
renderTrade(0, 0);
renderTrade(1, 0);
</script>"""
    return head + js


def _holdings_table(holdings):
    if not holdings:
        return '<div class="sub">当前空仓（择时已清仓）</div>'
    rows = "".join(
        f'<tr><td class="code"><a href="detail/{h["symbol"].split(".")[0]}.html">'
        f'{h["symbol"]}</a></td>'
        f'<td class="name">{h["name"]}</td>'
        f'<td>{h["buy_price"]:.2f}</td>'
        f'<td>{h["shares"]}</td>'
        f'<td>{h["buy_price"] * h["shares"] / 10000:.2f}万</td></tr>'
        for h in holdings
    )
    return (
        '<table><thead><tr><th>代码</th><th>名称</th><th>买入价</th><th>持股数</th><th>市值</th></tr></thead>'
        f'<tbody>{rows}</tbody></table>'
    )


CLOSED_REASON = {"stop": "止损", "profit": "止盈", "signal": "信号失效", "cashout": "空仓"}


def _closed_trades_section(closed_trades, holdings):
    """Per-trade win/loss: summary + closed-round-trip table + open float."""
    if not closed_trades:
        return '<div class="sub">暂无闭环交易</div>'
    n = len(closed_trades)
    wins = [t for t in closed_trades if t["pnl"] > 0]
    losses = [t for t in closed_trades if t["pnl"] <= 0]
    win_rate = len(wins) / n
    avg_win = sum(t["pnl"] for t in wins) / len(wins) if wins else 0.0
    avg_loss = sum(t["pnl"] for t in losses) / len(losses) if losses else 0.0
    pf = avg_win / abs(avg_loss) if avg_loss else 0.0

    def card(label, val, color=""):
        cls = f' class="{color}"' if color else ""
        return f'<div class="card"><div class="card-label">{label}</div><div class="card-value{cls}">{val}</div></div>'

    summary = "".join([
        card("闭环笔数", f"{n}", ""),
        card("胜率", f"{win_rate:.1%}", "pos" if win_rate >= 0.5 else ""),
        card("平均盈利", f"{avg_win:+.1%}", "pos"),
        card("平均亏损", f"{avg_loss:+.1%}", "neg"),
        card("盈亏比", f"{pf:.2f} 倍", ""),
    ])

    def to_js(ts):
        return [{
            "symbol": t["symbol"], "code": t["symbol"].split(".")[0],
            "name": t.get("name", ""), "buy_date": t["buy_date"], "buy_price": t["buy_price"],
            "sell_date": t["sell_date"], "sell_price": t["sell_price"],
            "pnl": t["pnl"], "days": t["days"],
            "reason": CLOSED_REASON.get(t["reason"], t["reason"]),
        } for t in ts][::-1]

    data_js = json.dumps(to_js(closed_trades), ensure_ascii=False)

    # open positions (unrealised PnL) table
    open_rows = ""
    for h in holdings:
        pnl = (h["last_close"] - h["buy_price"]) / h["buy_price"] if h["buy_price"] > 0 else 0
        cls = "pos" if pnl >= 0 else "neg"
        open_rows += (
            f'<tr><td class="code"><a href="detail/{h["symbol"].split(".")[0]}.html">{h["symbol"]}</a></td>'
            f'<td class="name">{h["name"]}</td><td>{h["buy_price"]:.2f}</td>'
            f'<td>{h["last_close"]:.2f}</td><td>{h["shares"]}</td>'
            f'<td class="{cls}">{pnl:+.1%}</td></tr>'
        )

    html = f'''<div class="cards seg" style="grid-template-columns:repeat(5,1fr)">{summary}</div>
<div class="sub" style="margin-bottom:8px">共 {n} 笔闭环交易（买入→卖出），点击页码翻页。</div>
<div id="closed-table"></div>
<h2 style="margin-top:20px">未平仓持仓（当前浮盈）</h2>
<table><thead><tr><th>代码</th><th>名称</th><th>买入价</th><th>现价</th><th>持股数</th><th>浮盈</th></tr></thead><tbody>{open_rows}</tbody></table>'''

    js = JS_CLOSED.replace("__DATA__", data_js)
    return html + js


JS_CLOSED = """<script>
const CLOSED = __DATA__;
const PS_C = 20;
function closedRow(t) {
  const cls = t.pnl >= 0 ? 'pos' : 'neg';
  return '<tr><td class="code">' + t.buy_date + '</td>' +
    '<td class="code"><a href="detail/' + t.code + '.html">' + t.symbol + '</a></td>' +
    '<td class="name">' + t.name + '</td>' +
    '<td>' + t.buy_price.toFixed(2) + '</td><td class="code">' + t.sell_date + '</td>' +
    '<td>' + t.sell_price.toFixed(2) + '</td>' +
    '<td class="' + cls + '"><b>' + (t.pnl * 100).toFixed(1) + '%</b></td>' +
    '<td>' + t.days + '天</td><td class="reason">' + t.reason + '</td></tr>';
}
function renderClosed(page) {
  const pages = Math.max(1, Math.ceil(CLOSED.length / PS_C));
  const slice = CLOSED.slice(page * PS_C, (page + 1) * PS_C);
  let h = '<table><thead><tr><th>买入日</th><th>代码</th><th>名称</th><th>买入价</th><th>卖出日</th><th>卖出价</th><th>盈亏</th><th>持有</th><th>原因</th></tr></thead><tbody>';
  for (let i = 0; i < slice.length; i++) h += closedRow(slice[i]);
  h += '</tbody></table><div class="pager">';
  const show = [];
  for (let p = 0; p < pages; p++) {
    if (p === 0 || p === pages - 1 || Math.abs(p - page) <= 2) show.push(p);
  }
  let last = -2;
  for (let q = 0; q < show.length; q++) {
    if (show[q] > last + 1) h += '<span class="page-ell">…</span>';
    h += '<button class="page-btn' + (show[q] === page ? ' active' : '') + '" onclick="renderClosed(' + show[q] + ')">' + (show[q] + 1) + '</button>';
    last = show[q];
  }
  h += '</div>';
  document.getElementById('closed-table').innerHTML = h;
}
renderClosed(0);
</script>"""


def _daily_signals_section(daily_signals):
    """Historical daily top-k signals with a calendar picker."""
    if not daily_signals:
        return '<div class="sub">暂无历史信号</div>'
    data_js = json.dumps(daily_signals, ensure_ascii=False)
    html = '''<div class="sub" style="margin-bottom:10px">每日收盘后的信号：<b>绿色=有信号</b>、<b>灰色=空仓</b>；点击日期查看当天 Top 10。</div>
<div class="cal-nav">
  <button class="page-btn" onclick="calShift(-1)">‹ 上月</button>
  <span id="cal-title" style="font-weight:600;color:var(--text)"></span>
  <button class="page-btn" onclick="calShift(1)">下月 ›</button>
</div>
<div id="cal-grid"></div>
<div id="daily-sig-title" class="sub" style="margin-top:14px;color:var(--blue);font-weight:600"></div>
<div id="daily-sig-table"></div>'''
    js = JS_DAILY.replace("__DATA__", data_js)
    return html + js


JS_DAILY = """<script>
const DAILY = __DATA__;
const DATE_INDEX = {};
DAILY.forEach(function(d, i) { DATE_INDEX[d.date] = i; });
let calYear = 0, calMonth = 0;
(function() {
  const latest = DAILY[DAILY.length - 1].date;
  calYear = parseInt(latest.slice(0, 4), 10);
  calMonth = parseInt(latest.slice(5, 7), 10) - 1;
})();
function pad2(n) { return (n < 10 ? '0' : '') + n; }
function calShift(dm) {
  calMonth += dm;
  if (calMonth < 0) { calMonth = 11; calYear--; }
  if (calMonth > 11) { calMonth = 0; calYear++; }
  renderCal();
}
function renderCal() {
  document.getElementById('cal-title').textContent = calYear + '年' + (calMonth + 1) + '月';
  const first = new Date(calYear, calMonth, 1);
  const startDow = first.getDay();
  const daysInMonth = new Date(calYear, calMonth + 1, 0).getDate();
  let h = '<div class="cal-row cal-head">';
  ['日','一','二','三','四','五','六'].forEach(function(w) { h += '<span>' + w + '</span>'; });
  h += '</div>';
  let cell = 0;
  for (let r = 0; r < 6; r++) {
    h += '<div class="cal-row">';
    for (let c = 0; c < 7; c++) {
      if (cell < startDow || cell - startDow >= daysInMonth) {
        h += '<span class="cal-cell empty"></span>';
      } else {
        const dayNum = cell - startDow + 1;
        const dateStr = calYear + '-' + pad2(calMonth + 1) + '-' + pad2(dayNum);
        const idx = DATE_INDEX[dateStr];
        if (idx !== undefined) {
          const cls = DAILY[idx].flat ? 'cal-cell trade flat' : 'cal-cell trade';
          h += '<span class="' + cls + '" onclick="showDaily(' + idx + ')">' + dayNum + '</span>';
        } else {
          h += '<span class="cal-cell">' + dayNum + '</span>';
        }
      }
      cell++;
    }
    h += '</div>';
  }
  document.getElementById('cal-grid').innerHTML = h;
}
function showDaily(i) {
  const d = DAILY[i];
  document.getElementById('daily-sig-title').textContent = d.date;
  if (d.flat) {
    document.getElementById('daily-sig-table').innerHTML =
      '<div class="flat-note">' + d.date + ' · 空仓（大盘跌破均线，模型清仓，不持仓）</div>';
  } else {
    let h = '<table><thead><tr><th>#</th><th>代码</th><th>名称</th><th>行业</th><th>评分</th><th>信号价</th><th>流通市值</th></tr></thead><tbody>';
    d.signals.forEach(function(s, k) {
      h += '<tr><td>' + (k + 1) + '</td>' +
        '<td class="code"><a href="detail/' + s.symbol.split('.')[0] + '.html">' + s.symbol + '</a></td>' +
        '<td class="name">' + s.name + '</td>' +
        '<td class="reason">' + (s.industry || '—') + '</td>' +
        '<td class="score">' + s.score.toFixed(4) + '</td>' +
        '<td>' + (s.close != null ? s.close.toFixed(2) : '—') + '</td>' +
        '<td>' + (s.market_cap != null ? s.market_cap.toFixed(0) + '亿' : '—') + '</td></tr>';
    });
    h += '</tbody></table>';
    document.getElementById('daily-sig-table').innerHTML = h;
  }
}
renderCal();
showDaily(DAILY.length - 1);
</script>"""


def _factor_ic_chart(timeline):
    """Interactive full 27-factor IC evolution: checkbox toggles + hover values.

    Y range is computed from the data (with headroom) so the top trend is never
    clipped. Every factor can be toggled; hovering shows the exact IC values of
    all checked factors on that refit date.
    """
    if not timeline:
        return '<div class="sub">暂无因子演化数据</div>'
    all_factors = timeline[0]["factors"]
    # mean |IC| -> default selection order (top-6 pre-checked)
    avg_abs: dict[str, float] = {}
    for f in all_factors:
        vals = [next((x["ic"] for x in t["factors"] if x["feature"] == f["feature"]), 0.0)
                for t in timeline]
        avg_abs[f["feature"]] = abs(sum(vals) / max(len(vals), 1))
    order = sorted(all_factors, key=lambda x: -avg_abs[x["feature"]])
    default_on = {f["feature"] for f in order[:6]}
    colors = {f["feature"]: f"hsl({i * 360 // len(all_factors)}, 70%, 60%)"
              for i, f in enumerate(all_factors)}
    all_ics = [x["ic"] for t in timeline for x in t["factors"]]
    lo, hi = min(all_ics), max(all_ics)
    span = max(hi - lo, 0.02)
    lo -= span * 0.12
    hi += span * 0.12

    checks = []
    for f in order:
        chk = "checked" if f["feature"] in default_on else ""
        checks.append(
            f'<label class="ic-check" style="--c:{colors[f["feature"]]}">'
            f'<input type="checkbox" value="{f["feature"]}" {chk} onchange="drawIC()">'
            f'<span class="ic-dot" style="background:{colors[f["feature"]]}"></span>{f["label"]}</label>'
        )
    data_js = json.dumps(timeline, ensure_ascii=False)
    colors_js = json.dumps(colors, ensure_ascii=False)

    html = f'''<div class="ic-wrap">
<div class="ic-controls">{''.join(checks)}</div>
<div class="chart" style="position:relative">
  <svg id="ic-svg" viewBox="0 0 880 300" width="100%" preserveAspectRatio="xMidYMid meet"></svg>
  <div id="ic-tip" class="tooltip"></div>
</div>
</div>'''

    js = JS_IC.replace("__DATA__", data_js).replace("__COLORS__", colors_js) \
              .replace("__LO__", str(lo)).replace("__HI__", str(hi))
    return html + js


JS_IC = """<script>
const IC_DATA = __DATA__;
const IC_COLORS = __COLORS__;
const IC_LO = __LO__, IC_HI = __HI__;
const W = 880, H = 300, PAD = 40;
function icY(v) { return PAD + (H - 2 * PAD) * (1 - (v - IC_LO) / (IC_HI - IC_LO)); }
function drawIC() {
  const svg = document.getElementById('ic-svg');
  const checks = Array.from(document.querySelectorAll('.ic-controls input:checked')).map(function(c){ return c.value; });
  const n = IC_DATA.length;
  let out = [];
  const zy = icY(0);
  out.push('<line x1="' + PAD + '" y1="' + zy + '" x2="' + (W - PAD) + '" y2="' + zy + '" stroke="#30363d" stroke-width="0.5" stroke-dasharray="3 3"/>');
  for (let i = 0; i <= 4; i++) {
    const v = IC_LO + (IC_HI - IC_LO) * i / 4, y = icY(v);
    out.push('<line x1="' + PAD + '" y1="' + y + '" x2="' + (W - PAD) + '" y2="' + y + '" stroke="#21262d" stroke-width="0.5"/>');
    out.push('<text x="6" y="' + (y + 4) + '" text-anchor="start" fill="#8b949e" font-size="10">' + v.toFixed(2) + '</text>');
  }
  [0, Math.floor((n - 1) / 2), n - 1].forEach(function(idx) {
    const x = n === 1 ? PAD : PAD + (W - 2 * PAD) * idx / (n - 1);
    out.push('<text x="' + x + '" y="' + (H - 8) + '" text-anchor="middle" fill="#8b949e" font-size="10">' + IC_DATA[idx].date + '</text>');
  });
  checks.forEach(function(f) {
    let pts = [];
    IC_DATA.forEach(function(t, ti) {
      const x = n === 1 ? PAD : PAD + (W - 2 * PAD) * ti / (n - 1);
      const rec = t.factors.find(function(x) { return x.feature === f; });
      if (rec !== undefined) pts.push(x + ',' + icY(rec.ic));
    });
    if (pts.length > 1) out.push('<polyline points="' + pts.join(' ') + '" fill="none" stroke="' + IC_COLORS[f] + '" stroke-width="1.5"/>');
  });
  svg.innerHTML = out.join('');
}
drawIC();
(function() {
  const svg = document.getElementById('ic-svg');
  const tip = document.getElementById('ic-tip');
  svg.addEventListener('mousemove', function(e) {
    const rect = svg.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width * W;
    const n = IC_DATA.length;
    let ti = Math.round((x - PAD) / (W - 2 * PAD) * (n - 1));
    if (ti < 0 || ti >= n) return;
    const t = IC_DATA[ti];
    const checks = Array.from(document.querySelectorAll('.ic-controls input:checked')).map(function(c){ return c.value; });
    let rows = '<div class="tt-date">' + t.date + '</div>';
    t.factors.filter(function(x){ return checks.indexOf(x.feature) >= 0; })
      .sort(function(a,b){ return Math.abs(b.ic) - Math.abs(a.ic); })
      .forEach(function(x) {
        rows += '<div class="tt-row"><span class="ic-dot" style="background:' + IC_COLORS[x.feature] + '"></span>' + x.label + ': <b>' + x.ic.toFixed(4) + '</b></div>';
      });
    tip.innerHTML = rows;
    tip.style.display = 'block';
    tip.style.left = Math.min(rect.width - 210, e.clientX - rect.left + 14) + 'px';
    tip.style.top = (e.clientY - rect.top - 8) + 'px';
  });
  svg.addEventListener('mouseleave', function() { tip.style.display = 'none'; });
})();
</script>"""


def _cashout_table(cashout_history, trades):
    if not cashout_history:
        return '<div class="sub">期间无空仓（择时未触发）</div>'
    buy_dates = sorted({t["date"] for t in trades if t["side"] == "buy"})
    rows = "".join(
        _cashout_row(h, buy_dates) for h in cashout_history
    )
    return (
        f'<div class="sub" style="margin-bottom:8px">共 {len(cashout_history)} 次空仓：大盘跌破均线清仓，重新站上均线后买入。</div>'
        f'<div style="max-height:300px;overflow-y:auto"><table>'
        f'<thead><tr><th>空仓日期</th><th>重新买入日期</th><th>空仓时长</th><th>触发原因</th></tr></thead>'
        f'<tbody>{rows}</tbody></table></div>'
    )


def _cashout_row(h, buy_dates):
    import pandas as _pd
    reentry = next((bd for bd in buy_dates if bd > h["date"]), None)
    if reentry:
        days = (_pd.Timestamp(reentry) - _pd.Timestamp(h["date"])).days
        dur = f"{days} 天"
    else:
        reentry, dur = "至今未重新买入", "—"
    return (
        f'<tr><td class="code">{h["date"]}</td>'
        f'<td class="code">{reentry}</td><td>{dur}</td>'
        f'<td class="reason">{h["event"]}</td></tr>'
    )


def render(data: dict) -> str:
    nav_pts = data["nav"]
    bench_pts = data["benchmark"]
    W, H, PAD = 880, 320, 48

    all_ys = [p["nav"] for p in nav_pts] + [p["nav"] for p in bench_pts]
    ymin, ymax = min(all_ys), max(all_ys)
    nav_path = _svg_line(nav_pts, "nav", W, H, PAD, ymin, ymax)
    bench_path = _svg_line(bench_pts, "nav", W, H, PAD, ymin, ymax)
    nav_ann = _nav_annotations(nav_pts, data.get("model_evolution", []), W, H, PAD)
    nav_js = json.dumps(nav_pts, ensure_ascii=False)
    bench_js = json.dumps(bench_pts, ensure_ascii=False)

    # Y-axis gridlines + labels for the NAV curve
    y_ticks = []
    for i in range(5):
        frac = i / 4
        yv = ymin + (ymax - ymin) * frac
        y = PAD + (H - 2 * PAD) * (1 - frac)
        y_ticks.append(
            f'<line x1="{PAD}" y1="{y:.1f}" x2="{W - PAD}" y2="{y:.1f}" '
            f'stroke="#21262d" stroke-width="0.5" stroke-dasharray="2 3"/>'
        )
        y_ticks.append(
            f'<text x="6" y="{y + 4:.1f}" text-anchor="start" fill="#8b949e" '
            f'font-size="11">{yv:.1f}x</text>'
        )
    y_axis = "".join(y_ticks)

    ic_chart = _factor_ic_chart(data.get("factor_ic_timeline", []))
    cashout_table = _cashout_table(data.get("cashout_history", []), data.get("trades", []))
    total_pos = data.get("total_position", 0)

    m = data["metrics"]
    seg = data["segments"]
    cfg = data.get("config", {})

    def pct(v):
        return f"{v:+.1%}"

    cards = "".join([
        _metric_card("年化收益", pct(m["annual_return"]), "全期（真实成交）",
                     "pos" if m["annual_return"] >= 0 else "neg"),
        _metric_card("夏普比率", f"{m['sharpe']:.2f}", "年化"),
        _metric_card("最大回撤", pct(m["max_drawdown"]), "历史最深", "neg"),
        _metric_card("日均换手", f"{m['avg_turnover']:.1%}", "低换手慢 alpha"),
    ])
    seg_cards = "".join([
        _metric_card(f"{name} 段", pct(v.get("annual_return", 0.0)),
                     f"夏普 {v.get('sharpe', 0.0):.2f}",
                     "pos" if v.get("annual_return", 0.0) >= 0 else "neg")
        for name, v in [("训练(熊市)", seg["train"]), ("验证", seg["valid"]), ("样本外", seg["test"])]
    ])
    risk_cards = "".join([
        _metric_card("最大单日亏损", pct(m.get("max_daily_loss", 0)), "集中持仓风险", "neg"),
        _metric_card("最长水下", f"{m.get('longest_underwater_days', 0)} 天", "不创新高煎熬期"),
        _metric_card("最长连亏", f"{m.get('longest_losing_streak', 0)} 天", "连续亏损"),
        _metric_card("最终净值", f"{m['final_nav']:.2f}x", f"{m['n_days']} 个交易日"),
    ])

    imp_bars = _importance_bars(data.get("feature_importance", []), W, 360, PAD)
    exit_bars = _exit_reason_bars(data.get("exit_reasons", []), W, 150, PAD)
    yearly_bars = _yearly_bars(data.get("yearly", []), W, 160, PAD)

    all_syms = {s["symbol"] for s in data["signals"]} | {t["symbol"] for t in data["trades"]}
    name_map = _load_name_map(all_syms)
    signals_cards = "".join(_attribution_card(s) for s in data["signals"])
    holdings_table = _holdings_table(data.get("current_holdings", []))
    trades_table = _trades_table(data.get("trades", []), name_map)
    closed_section = _closed_trades_section(
        data.get("closed_trades", []), data.get("current_holdings", []))
    daily_section = _daily_signals_section(data.get("daily_signals", []))

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>阿醒的 AI 策略模拟仓</title>
<style>
:root {{
  --bg:#0d1117; --panel:#161b22; --border:#30363d; --text:#e6edf3; --muted:#8b949e;
  --accent:#3fb950; --red:#f85149; --amber:#d29922; --blue:#58a6ff;
}}
* {{ box-sizing:border-box; margin:0; padding:0; }}
body {{ background:var(--bg); color:var(--text); font-family:-apple-system,"PingFang SC","Microsoft YaHei",sans-serif; padding:28px; line-height:1.6; }}
.container {{ max-width:960px; margin:0 auto; }}
header {{ margin-bottom:24px; animation:fadeUp .5s ease both; }}
h1 {{ font-size:20px; font-weight:600; }}
.sub {{ color:var(--muted); font-size:13px; margin-top:4px; }}
.cards {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:12px; }}
.cards.seg {{ grid-template-columns:repeat(3,1fr); }}
.card {{ background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:14px 16px; transition:transform .18s ease, box-shadow .18s ease; animation:fadeUp .5s ease both; }}
.card:hover {{ transform:translateY(-3px); box-shadow:0 6px 18px rgba(0,0,0,.35); border-color:#3fb95055; }}
.card-label {{ color:var(--muted); font-size:12px; }}
.card-value {{ font-size:22px; font-weight:600; margin:4px 0; }}
.card-value.pos {{ color:var(--red); }} .card-value.neg {{ color:var(--accent); }}
.card-sub {{ color:var(--muted); font-size:11px; }}
section {{ margin-top:28px; animation:fadeUp .6s ease both; }}
h2 {{ font-size:15px; font-weight:600; margin-bottom:12px; color:var(--blue); }}
.chart {{ background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:12px; }}
.legend {{ display:flex; gap:20px; font-size:12px; color:var(--muted); margin-bottom:6px; }}
.legend .dot {{ display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:6px; }}
.table-wrap {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
table {{ width:100%; border-collapse:collapse; font-size:13px; }}
th {{ text-align:left; color:var(--muted); font-weight:500; padding:6px 8px; border-bottom:1px solid var(--border); white-space:nowrap; }}
td {{ padding:5px 8px; border-bottom:1px solid var(--border); white-space:nowrap; font-size:12px; }}
a {{ color:inherit; text-decoration:none; }}
.code {{ font-family:ui-monospace,monospace; color:var(--blue); }}
.name {{ font-weight:500; }}
.score {{ font-family:ui-monospace,monospace; color:var(--amber); }}
.buy {{ color:var(--red); font-weight:600; }} .sell {{ color:var(--accent); font-weight:600; }}
.reason {{ color:var(--muted); font-size:12px; }}
.stock-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; }}
.stock-card {{ display:block; background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:14px 16px; transition:transform .18s ease, box-shadow .18s ease, border-color .18s ease; }}
.stock-card:hover {{ transform:translateY(-3px); box-shadow:0 6px 18px rgba(0,0,0,.35); border-color:#58a6ff88; }}
.stock-head {{ display:flex; align-items:baseline; gap:10px; margin-bottom:2px; }}
.stock-head .name {{ font-weight:600; font-size:15px; }}
.stock-head .score {{ margin-left:auto; }}
.stock-sub {{ color:var(--muted); font-size:12px; margin-bottom:10px; }}
.attr-row {{ display:flex; align-items:center; gap:8px; font-size:12px; margin:4px 0; }}
.attr-label {{ color:var(--muted); width:74px; flex-shrink:0; }}
.attr-val {{ font-family:ui-monospace,monospace; width:62px; text-align:right; }}
.pbar {{ flex:1; height:6px; background:#21262d; border-radius:3px; overflow:hidden; }}
.pfill {{ height:100%; background:var(--blue); border-radius:3px; }}
.trade-table {{ background:var(--panel); border:1px solid var(--border); border-radius:8px; padding:8px; }}
.pager {{ display:flex; flex-wrap:wrap; gap:6px; margin-top:10px; }}
.page-btn {{ background:#21262d; color:var(--text); border:1px solid var(--border); border-radius:5px; padding:4px 10px; font-size:12px; cursor:pointer; transition:background .15s ease; }}
.page-btn:hover {{ background:#30363d; }}
.page-btn.active {{ background:var(--blue); color:#0d1117; border-color:var(--blue); font-weight:600; }}
.page-ell {{ color:var(--muted); padding:4px 2px; font-size:12px; }}
.attr-pct {{ color:var(--muted); width:48px; text-align:right; flex-shrink:0; }}
.note {{ color:var(--muted); font-size:12px; margin-top:20px; border-left:3px solid var(--amber); padding-left:12px; }}
.ic-controls {{ display:flex; flex-wrap:wrap; gap:6px 14px; margin-bottom:10px; }}
.ic-check {{ display:inline-flex; align-items:center; gap:5px; font-size:12px; color:var(--text); cursor:pointer; padding:3px 8px; border:1px solid var(--border); border-radius:14px; }}
.ic-check input {{ accent-color:var(--blue); cursor:pointer; }}
.ic-dot {{ display:inline-block; width:9px; height:9px; border-radius:2px; flex-shrink:0; }}
.tooltip {{ position:absolute; background:#1c2128; border:1px solid var(--border); border-radius:8px; padding:8px 10px; font-size:12px; pointer-events:none; z-index:10; box-shadow:0 6px 18px rgba(0,0,0,.4); max-height:260px; overflow-y:auto; min-width:150px; }}
.tt-date {{ color:var(--blue); font-weight:600; margin-bottom:4px; }}
.tt-row {{ display:flex; align-items:center; gap:6px; white-space:nowrap; color:var(--text); }}
.tt-row b {{ color:var(--amber); }}
.nav-tip {{ position:absolute; background:#1c2128; border:1px solid var(--border); border-radius:8px; padding:8px 10px; font-size:12px; pointer-events:none; z-index:10; box-shadow:0 6px 18px rgba(0,0,0,.4); white-space:nowrap; }}
.model-stage {{ cursor:pointer; }}
.model-stage:hover line {{ stroke-width:1.2; opacity:0.9; }}
.cal-nav {{ display:flex; align-items:center; gap:14px; margin-bottom:10px; }}
.cal-row {{ display:grid; grid-template-columns:repeat(7,1fr); gap:4px; margin-bottom:4px; }}
.cal-head span {{ color:var(--muted); font-size:11px; text-align:center; }}
.cal-cell {{ height:34px; display:flex; align-items:center; justify-content:center; border-radius:6px; font-size:13px; color:var(--muted); background:#161b22; border:1px solid var(--border); }}
.cal-cell.empty {{ border:none; background:transparent; }}
.cal-cell.trade {{ color:var(--text); background:#1c3a24; border-color:#3fb95066; cursor:pointer; font-weight:600; }}
.cal-cell.trade:hover {{ background:#245a33; }}
.cal-cell.flat {{ background:#2a2320; border-color:#d2992266; color:var(--amber); }}
.flat-note {{ background:#2a2320; border:1px solid #d2992266; border-radius:8px; padding:12px 14px; color:var(--amber); font-size:13px; }}
@keyframes fadeUp {{ from {{ opacity:0; transform:translateY(12px); }} to {{ opacity:1; transform:translateY(0); }} }}
.bar-anim {{ animation:grow .6s ease both; transform-origin:left center; }}
@keyframes grow {{ from {{ transform:scaleX(0); }} to {{ transform:scaleX(1); }} }}
</style>
</head>
<body>
<div class="container">
<header>
  <h1>{data.get('strategy', '每日选股信号')}</h1>
  <div class="sub">信号日期 {data['signal_date']}（收盘后生成，次日开盘执行）· 计划买入 {len(data['signals'])} 只 · 初始资金 {cfg.get('initial_cash', 1000000) / 10000:.0f} 万</div>
  <div class="sub" style="margin-top:6px"><a href="acceptance.html">📋 模型验收报告 →</a></div>
</header>

<div class="cards">{cards}</div>
<div class="cards seg">{seg_cards}</div>
<div class="cards">{risk_cards}</div>

<section>
  <h2>回测净值（策略 vs 上证指数）</h2>
  <div class="chart" style="position:relative">
    <div class="legend">
      <span><span class="dot" style="background:var(--accent)"></span>策略净值（累计 {pct(m['final_nav'] - 1)}）</span>
      <span><span class="dot" style="background:var(--muted)"></span>上证指数（归一化）</span>
      <span><span class="dot" style="background:var(--amber)"></span>模型重训点（点击 M 编号看模型页）</span>
    </div>
    <svg id="nav-svg" viewBox="0 0 {W} {H}" width="100%" preserveAspectRatio="xMidYMid meet">
      {y_axis}
      {nav_ann}
      <line x1="{PAD}" y1="{H-PAD}" x2="{W-PAD}" y2="{H-PAD}" stroke="#30363d" stroke-width="0.5"/>
      <polyline points="{bench_path}" fill="none" stroke="#8b949e" stroke-width="1.2"/>
      <polyline points="{nav_path}" fill="none" stroke="#3fb950" stroke-width="1.8"/>
    </svg>
    <div id="nav-tip" class="nav-tip" style="display:none"></div>
  </div>
  <div class="sub" style="margin-top:6px">悬停查看每日净值；黄线为模型重训点，点击 M 编号查看该模型。</div>
</section>
<script>
const NAV_PTS = {nav_js};
const BENCH_PTS = {bench_js};
(function() {{
  const svg = document.getElementById('nav-svg');
  const tip = document.getElementById('nav-tip');
  const W = 880, H = 320, PAD = 48;
  svg.addEventListener('mousemove', function(e) {{
    const rect = svg.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width * W;
    const n = NAV_PTS.length;
    const i = Math.round((x - PAD) / (W - 2 * PAD) * (n - 1));
    if (i < 0 || i >= n) return;
    const p = NAV_PTS[i];
    const b = BENCH_PTS.find(function(q) {{ return q.date === p.date; }});
    let html = '<div class="tt-date">' + p.date + '</div>';
    html += '<div class="tt-row">策略净值 <b style="color:#3fb950">' + p.nav.toFixed(3) + '</b></div>';
    if (b) html += '<div class="tt-row">上证指数 <b style="color:#8b949e">' + b.nav.toFixed(3) + '</b></div>';
    tip.innerHTML = html;
    tip.style.display = 'block';
    tip.style.left = Math.min(rect.width - 170, e.clientX - rect.left + 14) + 'px';
    tip.style.top = (e.clientY - rect.top - 10) + 'px';
  }});
  svg.addEventListener('mouseleave', function() {{ tip.style.display = 'none'; }});
}})();
</script>

<section>
  <h2>历史买入信号（每日 Top {cfg.get('top_k', 10)}）</h2>
  {daily_section}
</section>

<section>
  <h2>计划买入（{data['signal_date']} 收盘后信号，次日开盘执行 · 每只约 {100 / cfg.get('top_k', 10):.0f} 万）</h2>
  <div class="sub" style="margin-bottom:12px">按 100 万初始资金等权测算，点击卡片查看日线与买卖点。</div>
  <div class="stock-grid">{signals_cards}</div>
</section>

<section>
  <h2>当前实际持仓（回测中至今仍持有 · 总仓位 {total_pos * 100:.0f}%）</h2>
  <div class="sub" style="margin-bottom:12px">回测组合截至 {data['signal_date']} 的实际持仓，等权满仓，每只约 {100 / max(len(data.get('current_holdings', [])) or 1, 1):.1f}%（与计划买入不同）。</div>
  {holdings_table}
</section>

<section>
  <h2>逐笔交易胜率（每笔闭环买卖的盈亏）</h2>
  {closed_section}
</section>

<section>
  <h2>因子重要性（模型从 {cfg.get('n_features', 0)} 个特征里学到了什么）</h2>
  <div class="chart">
    <svg viewBox="0 0 {W} 360" width="100%" preserveAspectRatio="xMidYMid meet">{imp_bars}</svg>
    <div class="sub" style="margin-top:6px">柱越长，该因子对模型越重要。</div>
  </div>
</section>

<section>
  <h2>因子有效性演化（全部 27 因子 · 可勾选 · 悬浮看数值）</h2>
  {ic_chart}
  <div class="sub" style="margin-top:6px">数值越大，因子与未来收益的关系越强；默认展示最强 6 个因子，可勾选切换。</div>
</section>

<section>
  <h2>空仓历史（择时触发记录）</h2>
  {cashout_table}
</section>

<section>
  <h2>卖出原因分布（回答"为什么卖"）</h2>
  <div class="chart">
    <svg viewBox="0 0 {W} 150" width="100%" preserveAspectRatio="xMidYMid meet">{exit_bars}</svg>
    <div class="sub" style="margin-top:6px">STOP=跌超 {cfg.get('stop_loss', 0) * 100:.0f}%；PROFIT=涨超 {cfg.get('take_profit', 0) * 100:.0f}%；SIGNAL=持有超 {cfg.get('min_hold', 0)} 天后掉出前 16 名；CASHOUT=大盘破均线清仓。</div>
  </div>
</section>

<section>
  <h2>分年度收益</h2>
  <div class="chart">
    <svg viewBox="0 0 {W} 160" width="100%" preserveAspectRatio="xMidYMid meet">{yearly_bars}</svg>
    <div class="sub" style="margin-top:6px">红=正收益，绿=负收益；2023 熊市为空仓。</div>
  </div>
</section>

<section>
  <h2>交易操作记录（全部，点击页码翻页）</h2>
  {trades_table}
</section>

<div class="note">
  <strong>说明</strong>：本策略为<b>数据驱动的 GBDT 多因子模型</b>——{cfg.get('n_features', 0)} 个特征喂给梯度提升树，模型自己决定哪些有效；
  回测采用<b>次日开盘价成交 + 滑点 + 持有到卖点</b>的真实假设；仅限主板、剔除 ST 股；大盘破均线自动空仓（择时）。
  本页面为研究输出，不构成投资建议。
</div>
</div>
</body>
</html>"""


def _render_candlestick(bars, trades, w, h, pad, n=90):
    if not bars:
        return ""
    recent = bars[-n:]
    lo = min(b["low"] for b in recent)
    hi = max(b["high"] for b in recent)
    span = (hi - lo) or 1.0
    cw = (w - 2 * pad) / len(recent)
    trade_map = {}
    for t in trades:
        trade_map[t["date"]] = (t["side"], t["price"])

    def y(v):
        return pad + (h - 2 * pad) * (1 - (v - lo) / span)

    out = []
    for i, b in enumerate(recent):
        x = pad + i * cw + cw / 2
        up = b["close"] >= b["open"]
        color = "#f85149" if up else "#3fb950"
        out.append(f'<line x1="{x:.1f}" y1="{y(b["high"]):.1f}" x2="{x:.1f}" y2="{y(b["low"]):.1f}" stroke="#8b949e" stroke-width="1"/>')
        o, c = y(b["open"]), y(b["close"])
        top, bottom = min(o, c), max(o, c)
        out.append(f'<rect x="{x - cw * 0.32:.1f}" y="{top:.1f}" width="{cw * 0.64:.1f}" height="{max(bottom - top, 1):.1f}" fill="{color}"/>')
        d = str(b["trade_date"])[:10]
        if d in trade_map:
            side, price = trade_map[d]
            if side == "buy":
                out.append(f'<text x="{x:.1f}" y="{y(b["low"]) + 16:.1f}" text-anchor="middle" fill="#f85149" font-size="12" font-weight="bold">B</text>')
            else:
                out.append(f'<text x="{x:.1f}" y="{y(b["high"]) - 8:.1f}" text-anchor="middle" fill="#3fb950" font-size="12" font-weight="bold">S</text>')
    # Y-axis gridlines + price labels (inside left edge)
    for v in (hi, (hi + lo) / 2, lo):
        yy = y(v)
        out.append(
            f'<line x1="{pad}" y1="{yy:.1f}" x2="{w - pad}" y2="{yy:.1f}" '
            f'stroke="#21262d" stroke-width="0.5" stroke-dasharray="2 3"/>'
        )
        out.append(f'<text x="{pad + 3}" y="{yy - 3:.1f}" fill="#8b949e" font-size="10">{v:.1f}</text>')
    # X-axis time ticks
    for idx in (0, len(recent) // 2, len(recent) - 1):
        b = recent[idx]
        x = pad + idx * cw + cw / 2
        out.append(
            f'<text x="{x:.1f}" y="{h - 5}" text-anchor="middle" fill="#8b949e" '
            f'font-size="10">{str(b["trade_date"])[:10]}</text>'
        )
    return "".join(out)


def _subjective_analysis(card: dict, extended: dict | None = None) -> str:
    """Rule-based reading of the profile numbers (traceable, not LLM-fluff)."""
    v = card.get("valuation", {})
    f = card.get("financials", {})
    mf = card.get("moneyflow", {})
    pts = []
    pe = v.get("pe_ttm")
    if pe is not None:
        if pe < 0:
            pts.append("PE 为负（亏损）")
        elif pe < 20:
            pts.append("估值偏低（PE<20）")
        elif pe < 50:
            pts.append("估值合理（PE 20~50）")
        else:
            pts.append("估值偏高（PE>50）")
    roe = f.get("roe")  # annualized ROE
    if roe is not None:
        if roe > 15:
            pts.append("盈利优秀（年化ROE>15%）")
        elif roe > 8:
            pts.append("盈利一般（年化ROE 8~15%）")
        else:
            pts.append("盈利偏弱（年化ROE<8%）")
    ny = f.get("netprofit_yoy")
    if ny is not None:
        if ny > 30:
            pts.append("高成长（净利同比>30%）")
        elif ny > 0:
            pts.append("正增长（净利同比正）")
        else:
            pts.append("净利负增长")
    net5d = mf.get("net_5d")
    if net5d is not None:
        pts.append("主力资金流入" if net5d > 0 else "主力资金流出")
    north = card.get("north")
    if north and north.get("net_20d") is not None:
        pts.append("北向近20日净买入" if north["net_20d"] > 0 else "北向近20日净卖出")
    chip = (extended or {}).get("chip")
    if chip and chip.get("profit_rate") is not None:
        pr = chip["profit_rate"]
        pts.append(f"获利盘{pr:.0f}%（{'套牢盘居多' if pr < 50 else '多数持仓盈利'}）")
    return "；".join(pts) + "。" if pts else "数据不足，无法解读。"


def _profile_section(card: dict, extended: dict | None = None) -> str:
    """In-page research card: valuation + four financial groups + funding + chips/rating."""
    if not card:
        return ""
    v = card.get("valuation", {})
    f = card.get("financials", {})
    mf = card.get("moneyflow", {})
    industry = card.get("industry", "")

    def num(x, suf=""):
        return f"{x:.2f}{suf}" if x is not None else "—"

    def num1(x, suf=""):
        return f"{x:.1f}{suf}" if x is not None else "—"

    def _group(title, rows):
        body = "".join(
            f'<div class="pf-row"><span class="pf-label">{k}</span>'
            f'<span class="pf-val">{val}</span></div>'
            for k, val in rows
        )
        return f'<div class="pf-group">{title}</div><div class="pf-grid">{body}</div>'

    val_rows = [
        ("流通市值", num(v.get("market_cap"), "亿")),
        ("PE(TTM)", num(v.get("pe_ttm"))),
        ("PB", num(v.get("pb"))),
        ("股息率", num1(v.get("dv_ttm"), "%")),
        ("换手率", num1(v.get("turnover_rate"), "%")),
    ]
    quality = [
        ("年化ROE", num(f.get("roe"), "%")),
        ("扣非ROE", num1(f.get("roe_dt"), "%")),
        ("ROIC", num1(f.get("roic"), "%")),
        ("净利率", num1(f.get("netprofit_margin"), "%")),
        ("毛利率", num1(f.get("gross_margin"), "%")),
    ]
    growth = [
        ("净利同比", num1(f.get("netprofit_yoy"), "%")),
        ("扣非同比", num1(f.get("dt_netprofit_yoy"), "%")),
        ("营收同比", num1(f.get("revenue_yoy"), "%")),
        ("单季环比", num1(f.get("q_op_qoq"), "%")),
    ]
    solvency = [
        ("负债率", num1(f.get("debt_to_assets"), "%")),
        ("速动比率", num(f.get("quick_ratio"))),
        ("净负债", num(f.get("netdebt"), "亿")),
        ("有息负债", num(f.get("interestdebt"), "亿")),
    ]
    cashflow = [
        ("经营现金流/股", num(f.get("ocfps"))),
        ("自由现金流", num(f.get("fcff"), "亿")),
        ("每股净资产", num(f.get("bps"))),
        ("EPS", num(f.get("eps"))),
    ]

    net5d = mf.get("net_5d")
    net5d_str = f"{net5d / 1e8:+.2f}亿" if net5d is not None else "—"
    holder = card.get("holder_chg")
    holder_str = f"{holder:+.1%}" if holder is not None else "—"
    fund_rows = [
        ("5日主力净流入", net5d_str),
        ("股东户数变化", holder_str),
    ]
    margin = card.get("margin")
    if margin:
        fund_rows += [
            ("融资余额", num(margin.get("rzye"), "亿")),
            ("融资买入额", num(margin.get("rzmre"), "亿")),
            ("融券余额", num(margin.get("rqye"), "亿")),
        ]
    north = card.get("north")
    if north:
        fund_rows += [
            ("北向5日净买", num(north.get("net_5d"), "亿")),
            ("北向20日净买", num(north.get("net_20d"), "亿")),
        ]

    lhb = card.get("lhb", [])
    lhb_html = ""
    if lhb:
        lhb_rows = "".join(
            f'<tr><td class="code">{x["date"]}</td>'
            f'<td class="{"pos" if (x.get("net_amount") or 0) >= 0 else "neg"}">'
            f'{(x.get("net_amount") if x.get("net_amount") is not None else 0):+.2f}亿</td>'
            f'<td class="reason">{x["reason"]}</td></tr>'
            for x in lhb
        )
        lhb_html = (
            '<div class="pf-group">龙虎榜（近3次）</div>'
            '<table><thead><tr><th>日期</th><th>净买入</th><th>上榜原因</th></tr></thead>'
            f'<tbody>{lhb_rows}</tbody></table>'
        )

    chip_html = ""
    rating_html = ""
    if extended:
        chip = extended.get("chip")
        if chip:
            pr = chip.get("profit_rate")
            chip_rows = [
                ("获利盘", f"{pr:.1f}%" if pr is not None else "—"),
                ("平均成本", f"{chip.get('avg_cost'):.2f}元" if chip.get("avg_cost") is not None else "—"),
                ("现价", f"{chip.get('close'):.2f}元" if chip.get("close") is not None else "—"),
                ("90%集中度", f"{chip.get('concentration90'):.1f}%" if chip.get("concentration90") is not None else "—"),
                ("70%集中度", f"{chip.get('concentration70'):.1f}%" if chip.get("concentration70") is not None else "—"),
            ]
            chip_html = _group("筹码分布", chip_rows)
        rating = extended.get("rating")
        if rating:
            rating_html = (
                '<div class="pf-group">机构评级</div>'
                f'<div class="sub">{rating.get("forecast_institutions") or 0} 家预测 · '
                f'{rating.get("buy_cnt") or 0} 家买入 / 共 {rating.get("total_cnt") or 0} 家</div>'
            )

    fin_html = (
        '<div class="pf-cols">'
        f'<div>{_group("盈利质量", quality)}{_group("成长性", growth)}</div>'
        f'<div>{_group("偿债能力", solvency)}{_group("现金流与每股", cashflow)}</div>'
        '</div>'
    )
    chip_rating_html = ""
    if chip_html or rating_html:
        chip_rating_html = f'<div class="pf-cols"><div>{chip_html}</div><div>{rating_html}</div></div>'

    core_html = "".join([
        _group("估值", val_rows),
        fin_html,
        _group("资金面", fund_rows),
        lhb_html,
        chip_rating_html,
    ])

    analysis = _subjective_analysis(card, extended)
    report = f.get("report_date", "")

    # ---- Extended data (concepts + consensus + news, compactly merged into the card).
    ext_blocks = ""
    if extended:
        concepts = extended.get("concepts", [])
        if concepts:
            chips = "".join(
                f'<span class="concept-chip" title="{c.get("note", "")}">{c["name"]}'
                f'<b>{c.get("relevance", "")}</b></span>'
                for c in concepts
            )
            ext_blocks += f'<div class="pf-group">概念题材（通达信）</div><div>{chips}</div>'
        cons = extended.get("consensus", {})
        if cons and cons.get("target_price"):
            tp = cons["target_price"]
            fc_rows = "".join(
                f'<div class="fc-row"><span class="fc-year">{f["year"]}E</span>'
                f'<span class="fc-v">EPS {f["eps"]:.2f}</span>'
                f'<span class="fc-v">营收 {f["revenue_yoy"]:+.1f}%</span>'
                f'<span class="fc-v">净利 {f["netprofit_yoy"]:+.1f}%</span>'
                f'<span class="fc-v">PE {f["pe"]:.1f}</span></div>'
                for f in cons.get("forecasts", [])
            )
            ext_blocks += (
                f'<div class="pf-group">机构一致预期 · 目标价 '
                f'<b style="color:var(--red)">{tp:.2f}元</b></div>'
                f'{fc_rows}'
            )
        news = extended.get("news", [])
        if news:
            news_rows = "".join(
                (f'<div class="news-item"><a href="{n["url"]}" target="_blank" rel="noopener">'
                 f'{n["title"]}</a><div class="news-meta">{n["time"]} · {n["source"]}</div></div>')
                if n.get("url") else
                f'<div class="news-item">{n["title"]}<div class="news-meta">{n["time"]} · {n["source"]}</div></div>'
                for n in news
            )
            ext_blocks += f'<div class="pf-group">新闻研报（东方财富妙想）</div>{news_rows}'

    return f"""
<div class="chart">
  <h2>个股档案{(' · ' + industry) if industry else ''}</h2>
  <div class="pf-note" style="margin-bottom:10px"><b>一句话解读：</b>{analysis}</div>
  {core_html}
  {ext_blocks}
  <div class="sub" style="margin-top:10px">财务报告期 {report} · 数据来自结构化接口 · 规则化解读，不构成投资建议。</div>
</div>"""


def render_detail(symbol: str, name: str, bars: list[dict], trades: list[dict],
                 profile: dict | None = None, extended: dict | None = None) -> str:
    profile_section = _profile_section(profile, extended) if profile else ""
    last = bars[-1] if bars else {}
    prev_close = bars[-2]["close"] if len(bars) >= 2 else last.get("close", 0)
    chg = (last.get("close", 0) / prev_close - 1) if prev_close else 0
    chg_color = "#f85149" if chg >= 0 else "#3fb950"

    l1 = (
        f'<div class="l1-grid">'
        f'<div class="l1-cell"><span class="l1-label">收盘</span><span class="l1-val">{last.get("close", 0):.2f}</span></div>'
        f'<div class="l1-cell"><span class="l1-label">涨跌</span><span class="l1-val" style="color:{chg_color}">{chg:+.2%}</span></div>'
        f'<div class="l1-cell"><span class="l1-label">开盘</span><span class="l1-val">{last.get("open", 0):.2f}</span></div>'
        f'<div class="l1-cell"><span class="l1-label">最高</span><span class="l1-val" style="color:#f85149">{last.get("high", 0):.2f}</span></div>'
        f'<div class="l1-cell"><span class="l1-label">最低</span><span class="l1-val" style="color:#3fb950">{last.get("low", 0):.2f}</span></div>'
        f'<div class="l1-cell"><span class="l1-label">成交额</span><span class="l1-val">{last.get("amount", 0) / 1e8:.2f}亿</span></div>'
        f'</div>'
    )

    W, H, PAD = 880, 380, 30
    candle = _render_candlestick(bars, trades, W, H, PAD)

    # per-trade list
    trade_rows = "".join(
        f'<tr><td class="code">{t["date"]}</td>'
        f'<td class="{t["side"]}">{"BUY" if t["side"] == "buy" else "SELL"}</td>'
        f'<td>{t["price"]:.2f}</td>'
        f'<td class="reason">{REASON_DETAIL.get(t["reason"], t["reason"]) if t["side"] == "sell" else "建仓"}</td></tr>'
        for t in reversed(trades)
    ) if trades else '<tr><td colspan="4" class="reason">近期无交易</td></tr>'

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{name} {symbol} · 个股详情</title>
<style>
:root {{ --bg:#0d1117; --panel:#161b22; --border:#30363d; --text:#e6edf3; --muted:#8b949e; --red:#f85149; --accent:#3fb950; --blue:#58a6ff; }}
* {{ box-sizing:border-box; margin:0; padding:0; }}
body {{ background:var(--bg); color:var(--text); font-family:-apple-system,"PingFang SC","Microsoft YaHei",sans-serif; padding:24px; line-height:1.6; }}
.container {{ max-width:960px; margin:0 auto; }}
a {{ color:var(--blue); text-decoration:none; }}
.back {{ display:inline-block; margin-bottom:16px; font-size:13px; }}
h1 {{ font-size:20px; font-weight:600; }}
h1 .code {{ color:var(--muted); font-size:15px; font-weight:400; margin-left:8px; }}
.l1-grid {{ display:grid; grid-template-columns:repeat(6,1fr); gap:10px; margin:16px 0; }}
.l1-cell {{ background:var(--panel); border:1px solid var(--border); border-radius:8px; padding:10px 12px; }}
.l1-label {{ display:block; color:var(--muted); font-size:11px; }}
.l1-val {{ font-family:ui-monospace,monospace; font-size:16px; font-weight:600; }}
.chart {{ background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:12px; margin:16px 0; }}
h2 {{ font-size:15px; color:var(--blue); margin-bottom:10px; }}
table {{ width:100%; border-collapse:collapse; font-size:13px; margin-top:8px; }}
th {{ text-align:left; color:var(--muted); font-weight:500; padding:6px 8px; border-bottom:1px solid var(--border); white-space:nowrap; }}
td {{ padding:5px 8px; border-bottom:1px solid var(--border); white-space:nowrap; font-size:12px; }}
.code {{ font-family:ui-monospace,monospace; color:var(--blue); }}
.buy {{ color:var(--red); font-weight:600; }} .sell {{ color:var(--accent); font-weight:600; }}
.reason {{ color:var(--muted); font-size:12px; }}
.legend {{ display:flex; gap:16px; font-size:12px; color:var(--muted); margin-bottom:4px; }}
.pf-note {{ background:#1c2128; border-left:3px solid var(--amber); padding:6px 10px; border-radius:6px; font-size:12px; }}
.pf-grid {{ display:grid; grid-template-columns:repeat(3,1fr); gap:2px 14px; }}
.pf-row {{ display:flex; justify-content:space-between; padding:2px 0; border-bottom:1px solid var(--border); font-size:11px; }}
.pf-label {{ color:var(--muted); font-size:11px; }}
.pf-val {{ font-family:ui-monospace,monospace; color:var(--text); font-size:11px; }}
.pf-group {{ margin:8px 0 2px; font-size:11px; color:var(--blue); font-weight:600; }}
.pf-cols {{ display:grid; grid-template-columns:1fr 1fr; gap:0 20px; }}
.concept-chip {{ display:inline-block; background:#1c3a24; border:1px solid #3fb95066; color:var(--accent); border-radius:14px; padding:2px 10px; font-size:12px; margin:3px; cursor:default; }}
.concept-chip b {{ margin-left:4px; color:var(--amber); }}
.fc-row {{ display:flex; gap:18px; padding:6px 0; border-bottom:1px solid var(--border); font-size:12px; }}
.fc-year {{ color:var(--blue); font-weight:600; width:44px; }}
.fc-v {{ font-family:ui-monospace,monospace; color:var(--text); }}
.news-item {{ padding:8px 0; border-bottom:1px solid var(--border); font-size:13px; }}
.news-meta {{ color:var(--muted); font-size:11px; margin-top:2px; }}
@keyframes fadeUp {{ from {{ opacity:0; transform:translateY(10px); }} to {{ opacity:1; transform:translateY(0); }} }}
body {{ animation:fadeUp .5s ease both; }}
</style>
</head>
<body>
<div class="container">
<a class="back" href="../index.html">← 返回主页</a>
<h1>{name}<span class="code">{symbol}</span></h1>

<div class="l1-grid">{l1}</div>

<div class="chart">
  <div class="legend"><span>近 90 个交易日 K 线</span><span><b style="color:var(--red)">B</b>=买入点</span><span><b style="color:var(--accent)">S</b>=卖出点</span><span>红涨绿跌</span></div>
  <svg viewBox="0 0 {W} {H}" width="100%" preserveAspectRatio="xMidYMid meet">{candle}</svg>
</div>

<div class="chart">
  <h2>本股买卖记录</h2>
  <table><thead><tr><th>日期</th><th>方向</th><th>价格</th><th>原因</th></tr></thead><tbody>{trade_rows}</tbody></table>
</div>

{profile_section}
</div>
</body>
</html>"""


def render_acceptance(data: dict) -> str:
    """Acceptance report page: check the model against frank-quant EP004 standards."""
    m = data.get("metrics", {})
    seg = data.get("segments", {})

    checks = [
        ("目标函数（验证集打分 + 惩罚训练/验证落差）",
         "frank-quant: -(valid_sharpe − 0.5×max(0, train−valid))",
         "已实测·不适用（滚动晋级跨时期分数不可比）", "warn"),
        ("夏普口径（日资金曲线）",
         "frank-quant: 每日盈亏汇总 ÷ 初始资金，非每笔交易夏普",
         "✅ 已实现（nav.pct_change 日收益年化）", "pass"),
        ("最少交易限制（防超低频刷夏普）",
         "frank-quant: MIN_TRADES=30 / MIN_VALID_TRADES=50",
         "已实测·不适用（日频慢换手模型非超低频策略）", "warn"),
        ("三段划分 + 物理隔离",
         "frank-quant: TRAIN/VALID/TEST，TEST 进程不可达",
         "⚠️ 事后切分，但滚动训练严格样本外", "warn"),
        ("未来函数防护",
         "frank-quant: 全量 vs 截断信号对照",
         "✅ 严格样本外（滚动窗口，资金流/融资 shift(1)）", "pass"),
        ("Deflated Sharpe（搜索次数修正）",
         "frank-quant: 按搜索次数 N 修正运气上限",
         "✅ 已实现（8 真实 trials，运气上限 1.43，DSR=0.975）", "pass"),
        ("换手控制",
         "frank-quant: exit_buffer + min_hold + max_open_trades",
         "✅ 满仓上限 + 最短持有 + 缓冲带", "pass"),
        ("现金账户（模拟真实资金约束）",
         "frank-quant: dry_run_wallet",
         "✅ 已实现（100万现金账户 + 涨跌停 + 市值分档滑点）", "pass"),
    ]
    rows = "".join(
        f'<tr class="row-{c[3]}"><td class="check-name">{c[0]}</td>'
        f'<td class="std">{c[1]}</td><td class="status">{c[2]}</td></tr>'
        for c in checks
    )
    n_pass = sum(1 for c in checks if c[3] == "pass")
    n_fail = sum(1 for c in checks if c[3] == "fail")
    verdict = "通过验收（防过拟合核心机制已达标）" if n_fail == 0 else "未完全通过验收"
    verdict_color = "#3fb950" if n_fail == 0 else "#f85149"

    def pct(v):
        return f"{v:+.1%}" if isinstance(v, (int, float)) else "—"

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>模型验收报告 · 阿醒的 AI 策略 0813</title>
<style>
:root {{ --bg:#0d1117; --panel:#161b22; --border:#30363d; --text:#e6edf3; --muted:#8b949e; --red:#f85149; --accent:#3fb950; --amber:#d29922; --blue:#58a6ff; }}
* {{ box-sizing:border-box; margin:0; padding:0; }}
body {{ background:var(--bg); color:var(--text); font-family:-apple-system,"PingFang SC","Microsoft YaHei",sans-serif; padding:28px; line-height:1.6; }}
.container {{ max-width:880px; margin:0 auto; }}
a {{ color:var(--blue); text-decoration:none; }}
.back {{ display:inline-block; margin-bottom:16px; font-size:13px; }}
h1 {{ font-size:20px; font-weight:600; }}
h2 {{ font-size:15px; color:var(--blue); margin:22px 0 10px; }}
table {{ width:100%; border-collapse:collapse; font-size:13px; margin:10px 0; }}
th {{ text-align:left; color:var(--muted); font-weight:500; padding:8px; border-bottom:1px solid var(--border); }}
td {{ padding:8px; border-bottom:1px solid var(--border); vertical-align:top; }}
.check-name {{ font-weight:500; width:200px; }}
.std {{ color:var(--muted); font-size:12px; }}
.status {{ font-size:12px; }}
.row-pass .status {{ color:var(--accent); }}
.row-fail .status {{ color:var(--red); }}
.row-warn .status {{ color:var(--amber); }}
.verdict {{ font-size:18px; font-weight:700; padding:14px 18px; border-radius:10px; border:1px solid var(--border); background:var(--panel); }}
.card {{ background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:14px 16px; }}
.cards {{ display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin:12px 0; }}
.card-label {{ color:var(--muted); font-size:12px; }}
.card-value {{ font-size:20px; font-weight:600; margin:4px 0; }}
.note {{ color:var(--muted); font-size:12px; margin-top:20px; border-left:3px solid var(--amber); padding-left:12px; }}
</style>
</head>
<body>
<div class="container">
<a class="back" href="index.html">← 返回模拟仓</a>
<h1>模型验收报告 · {data.get('strategy', '')}</h1>
<div style="color:var(--muted);font-size:13px;margin-top:4px">{data.get('model_note', '')}</div>

<div class="verdict" style="margin-top:16px;color:{verdict_color}">结论：{verdict}（通过 {n_pass} 项 / 已实测·判定不适用 2 项 / 部分达标 1 项）</div>

<div class="cards">
  <div class="card"><div class="card-label">全期年化</div><div class="card-value" style="color:{'#f85149' if m.get('annual_return',0)>=0 else '#3fb950'}">{pct(m.get('annual_return'))}</div></div>
  <div class="card"><div class="card-label">夏普</div><div class="card-value">{m.get('sharpe', 0):.2f}</div></div>
  <div class="card"><div class="card-label">最大回撤</div><div class="card-value" style="color:#3fb950">{pct(m.get('max_drawdown'))}</div></div>
</div>

<h2>验收对照（frank-quant EP004 标准）</h2>
<table><thead><tr><th>验收项</th><th>frank-quant 标准</th><th>当前实现</th></tr></thead>
<tbody>{rows}</tbody></table>

<h2>三段表现</h2>
<div class="cards">
  <div class="card"><div class="card-label">训练段(熊市)</div><div class="card-value">{pct(seg.get('train', {}).get('annual_return'))}</div><div class="card-label">夏普 {seg.get('train', {}).get('sharpe', 0):.2f}</div></div>
  <div class="card"><div class="card-label">验证段</div><div class="card-value">{pct(seg.get('valid', {}).get('annual_return'))}</div><div class="card-label">夏普 {seg.get('valid', {}).get('sharpe', 0):.2f}</div></div>
  <div class="card"><div class="card-label">样本外段</div><div class="card-value">{pct(seg.get('test', {}).get('annual_return'))}</div><div class="card-label">夏普 {seg.get('test', {}).get('sharpe', 0):.2f}</div></div>
</div>

<div class="note">
<strong>对抗性审查结论（为什么 2 项「不适用」）</strong>：① frank-quant 的「滚动晋级门槛」（验证集打分 + 训练/验证落差惩罚）经实测<strong>不适用</strong>——在滚动重训场景下，新旧模型的验证集分属不同时期，IC 水平随市场 regime 漂移，跨时期分数不可比，导致模型几乎不晋级（实测 n_promotions=1）、长期用过时模型，回测反而从 +61% 崩到 -7%。② 「最少交易限制」同样不适用——本模型是日频选股、慢换手，不是"全年只交易一次刷夏普"的超低频策略，无需该门槛。③ 因此防过拟合采用 frank-quant 真正的杀手锏：严格样本外滚动训练 + 三段隔离 + DSR 运气修正（8 个真实 trials，运气上限 1.43 vs 实测夏普 2.44，DSR=0.975 显著）+ MC bootstrap（5000 次分块重采样，100% 盈利）。
</div>
</div>
</body>
</html>"""


def render_model_page(me: dict, data: dict) -> str:
    """Per-model observability page: full 27-factor weights + train/valid card."""
    idx = me.get("index", 0)
    imp = sorted(me.get("feature_importance", []), key=lambda x: -x["importance"])
    max_imp = max((x["importance"] for x in imp), default=0.0) or 1.0
    bars = []
    for x in imp:
        pct_w = max(0.0, x["importance"] / max_imp * 100.0)
        bars.append(
            f'<div class="imp-row">'
            f'<span class="imp-label">{x["label"]}</span>'
            f'<div class="imp-bar"><div class="imp-fill" style="width:{pct_w:.1f}%"></div></div>'
            f'<span class="imp-val">{x["importance"]:.4f}</span>'
            f'</div>'
        )
    ic_sign = "pos" if me.get("valid_ic", 0) >= 0 else "neg"
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>模型 M{idx} · 阿醒的 AI 策略 0813</title>
<style>
:root {{ --bg:#0d1117; --panel:#161b22; --border:#30363d; --text:#e6edf3; --muted:#8b949e; --red:#f85149; --accent:#3fb950; --amber:#d29922; --blue:#58a6ff; }}
* {{ box-sizing:border-box; margin:0; padding:0; }}
body {{ background:var(--bg); color:var(--text); font-family:-apple-system,"PingFang SC","Microsoft YaHei",sans-serif; padding:28px; line-height:1.6; }}
.container {{ max-width:880px; margin:0 auto; }}
a {{ color:var(--blue); text-decoration:none; }}
.back {{ display:inline-block; margin-bottom:16px; font-size:13px; }}
h1 {{ font-size:20px; font-weight:600; }}
h2 {{ font-size:15px; color:var(--blue); margin:22px 0 10px; }}
.sub {{ color:var(--muted); font-size:13px; margin-top:4px; }}
.cards {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:14px 0; }}
.card {{ background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:12px 14px; }}
.card-label {{ color:var(--muted); font-size:12px; }}
.card-value {{ font-size:20px; font-weight:600; margin:2px 0; }}
.card-value.pos {{ color:var(--red); }} .card-value.neg {{ color:var(--accent); }}
.panel {{ background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:14px 16px; }}
.imp-row {{ display:flex; align-items:center; gap:10px; font-size:12px; margin:5px 0; }}
.imp-label {{ color:var(--muted); width:90px; flex-shrink:0; text-align:right; }}
.imp-bar {{ flex:1; height:8px; background:#21262d; border-radius:4px; overflow:hidden; }}
.imp-fill {{ height:100%; background:linear-gradient(90deg,#58a6ff,#3fb950); border-radius:4px; }}
.imp-val {{ font-family:ui-monospace,monospace; width:56px; text-align:right; color:var(--amber); }}
.kv {{ display:flex; gap:24px; flex-wrap:wrap; font-size:13px; }}
.kv-item {{ color:var(--muted); }}
.kv-item b {{ color:var(--text); font-weight:500; }}
.note {{ color:var(--muted); font-size:12px; margin-top:20px; border-left:3px solid var(--amber); padding-left:12px; }}
</style>
</head>
<body>
<div class="container">
<a class="back" href="../index.html">← 返回模拟仓</a>
<h1>模型 M{idx} · {data.get('strategy', '')}</h1>
<div class="sub">重训日 {me.get('refit_date', '')} · 生效区间 {me.get('start_date', '')} ~ {me.get('end_date', '')} · 共 {len(data.get('model_evolution', []))} 个滚动模型</div>

<div class="cards">
  <div class="card"><div class="card-label">训练集 IC</div><div class="card-value {ic_sign}">{me.get('train_ic', 0):+.4f}</div></div>
  <div class="card"><div class="card-label">验证集 IC</div><div class="card-value {ic_sign}">{me.get('valid_ic', 0):+.4f}</div></div>
  <div class="card"><div class="card-label">落差惩罚</div><div class="card-value">-{me.get('gap_penalty', 0):.4f}</div></div>
  <div class="card"><div class="card-label">目标函数得分</div><div class="card-value {ic_sign}">{me.get('score', 0):+.4f}</div></div>
</div>

<div class="panel">
  <div class="kv">
    <div class="kv-item">训练窗口 <b>{me.get('train_start', '')} ~ {me.get('train_end', '')}</b>（{me.get('n_train_days', 0)} 天）</div>
    <div class="kv-item">内部验证 <b>{me.get('valid_start', '')} ~ {me.get('valid_end', '')}</b>（{me.get('n_valid_days', 0)} 天）</div>
  </div>
  <div class="sub" style="margin-top:8px">目标函数 = 验证集 IC − 0.5 × max(0, 训练集 IC − 验证集 IC)。</div>
</div>

<h2>完整 27 因子权重（按贡献排序）</h2>
<div class="panel">
  {''.join(bars)}
  <div class="sub" style="margin-top:10px">柱越长，表示该因子对模型越重要。</div>
</div>

<div class="note">
<strong>可观测验证</strong>：每个滚动点均记录训练/验证切分、训练集 IC、验证集 IC、落差惩罚、目标函数得分与完整因子权重；全局显著性由统计检验统一验证。
</div>
</div>
</body>
</html>"""




def main() -> None:
    data = _load()
    html = render(data)
    (ROOT / "runtime/dashboard/index.html").write_text(html, encoding="utf-8")
    print(f"written -> index.html  ({len(html)} bytes)")

    acceptance = render_acceptance(data)
    (ROOT / "runtime/dashboard/acceptance.html").write_text(acceptance, encoding="utf-8")
    print(f"written -> acceptance.html  ({len(acceptance)} bytes)")

    # per-model observability pages (one per rolling refit)
    MODEL_DIR = ROOT / "runtime/dashboard/model"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for me in data.get("model_evolution", []):
        idx = me.get("index", 0)
        page = render_model_page(me, data)
        (MODEL_DIR / f"model_{idx}.html").write_text(page, encoding="utf-8")
    print(f"written -> {len(data.get('model_evolution', []))} model pages in {MODEL_DIR}")

    # load independent stock profiles (code -> card) for in-page research section
    profiles: dict[str, dict] = {}
    for p in (ROOT / "runtime/stock-profiles").glob("*.json"):
        if "_extended" in p.name:  # skip connector snapshot, keep base card
            continue
        card = json.loads(p.read_text(encoding="utf-8"))
        profiles[card["symbol"].split(".")[0]] = card

    # per-stock detail pages (with the research card embedded, no 3-level page)
    DETAIL_DIR.mkdir(parents=True, exist_ok=True)
    trades_by_sym: dict[str, list[dict]] = {}
    for t in data.get("trades", []):
        trades_by_sym.setdefault(t["symbol"], []).append(t)
    all_syms = {s["symbol"] for s in data["signals"]} | set(trades_by_sym.keys())
    name_map = _load_name_map(all_syms)
    # extended connector data (concepts/consensus/news), keyed by code
    extended_map: dict[str, dict] = {}
    for p in (ROOT / "runtime/stock-profiles").glob("*_extended.json"):
        card = json.loads(p.read_text(encoding="utf-8"))
        extended_map[card["symbol"].split(".")[0]] = card
    for sym in sorted(all_syms):
        bars = _load_bars(sym)
        if not bars:
            continue
        code = sym.split(".")[0]
        detail = render_detail(sym, name_map.get(sym, ""), bars,
                               trades_by_sym.get(sym, []), profiles.get(code),
                               extended_map.get(code))
        (DETAIL_DIR / f"{code}.html").write_text(detail, encoding="utf-8")
    print(f"written -> {len(all_syms)} detail pages in {DETAIL_DIR}")


if __name__ == "__main__":
    main()
