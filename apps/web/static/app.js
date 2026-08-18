"use strict";

const byId = (id) => document.getElementById(id);
const text = (value) => String(value ?? "—");
const money = (value) => Number(value).toLocaleString("zh-CN", {style: "currency", currency: "CNY"});
const percent = (value) => `${(Number(value) * 100).toFixed(2)}%`;

function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

function element(tag, value, className) {
  const node = document.createElement(tag);
  if (value !== undefined) node.textContent = text(value);
  if (className) node.className = className;
  return node;
}

function emptyRow(body, columns, message) {
  const row = element("tr");
  const cell = element("td", message, "muted");
  cell.colSpan = columns;
  row.appendChild(cell);
  body.appendChild(row);
}

function cards(target, items) {
  clear(target);
  for (const [label, value] of items) {
    const card = element("div", undefined, "card");
    card.append(element("div", label, "label"), element("div", value, "value"));
    target.appendChild(card);
  }
}

function renderSignal(signal) {
  const badge = byId("state-badge");
  badge.textContent = signal.state;
  badge.className = `badge ${signal.state.toLowerCase()}`;
  byId("as-of").textContent = `决策日 ${signal.as_of} · 完整数据截至 ${signal.latest_complete_date}`;
  cards(byId("signal-cards"), [
    ["信号序号", signal.sequence],
    ["状态", signal.state],
    ["策略", signal.champion ? `${signal.champion.strategy_id}@${signal.champion.strategy_version}` : "无活跃 Champion"],
    ["生成时间", signal.generated_at],
  ]);
  const body = byId("targets");
  clear(body);
  if (!signal.target_positions.length) emptyRow(body, 2, "空目标仓位");
  for (const target of signal.target_positions) {
    const row = element("tr");
    row.append(element("td", target.symbol), element("td", percent(target.target_weight), "right"));
    body.appendChild(row);
  }
  byId("reason-codes").textContent = `状态原因：${signal.reason_codes.join(" · ")}`;
}

function renderAccount(account) {
  byId("account-id").textContent = `${account.account_id} · 第 ${account.account_sequence} 期 · ${account.as_of}`;
  cards(byId("account-cards"), [
    ["初始资金", money(account.initial_cash)], ["模拟总资产", money(account.total_assets)],
    ["现金", money(account.cash)], ["模拟持仓市值", money(account.market_value)],
    ["净值", Number(account.nav).toFixed(6)],
  ]);
  const holdings = byId("holdings");
  clear(holdings);
  if (!account.holdings.length) emptyRow(holdings, 8, "模拟账户空仓");
  for (const holding of account.holdings) {
    const row = element("tr");
    const values = [
      [holding.symbol, ""], [holding.shares, "right"], [holding.locked_shares, "right"],
      [Number(holding.avg_cost).toFixed(3), "right"], [Number(holding.last_price).toFixed(3), "right"],
      [money(holding.market_value), "right"], [percent(holding.target_weight), "right"],
      [holding.valuation_frozen ? "估值冻结" : "当日估值", holding.valuation_frozen ? "muted" : ""],
    ];
    for (const [value, cls] of values) row.appendChild(element("td", value, cls));
    holdings.appendChild(row);
  }
  const trades = byId("trades");
  clear(trades);
  if (!account.trades.length) emptyRow(trades, 7, "当日无模拟成交");
  for (const trade of account.trades) {
    const row = element("tr");
    row.append(
      element("td", trade.trade_date), element("td", trade.symbol),
      element("td", trade.side === "buy" ? "模拟买入" : "模拟卖出", trade.side),
      element("td", trade.shares, "right"), element("td", Number(trade.price).toFixed(3), "right"),
      element("td", money(trade.total_cost), "right"), element("td", trade.reason),
    );
    trades.appendChild(row);
  }
  const skips = byId("skips");
  clear(skips);
  for (const skip of account.skips) skips.appendChild(element("li", `${skip.symbol} · ${skip.side} · ${skip.reason_code}`));
}

function renderProvenance(manifest, signal, account) {
  const list = byId("provenance");
  clear(list);
  const items = [
    ["Web 发布", manifest.release_id], ["发布时间", manifest.generated_at],
    ["数据集", signal.contract_set.dataset_manifest_sha256], ["数据快照", signal.contract_set.dataset_snapshot_sha256],
    ["Champion", signal.contract_set.champion_sha256 || "无"], ["信号哈希", manifest.signal_sha256],
    ["模拟账户状态", account.state_id], ["账户状态哈希", manifest.account_state_sha256],
  ];
  for (const [label, value] of items) list.append(element("dt", label), element("dd", value));
}

async function loadJson(path) {
  const response = await fetch(path, {cache: "no-store"});
  if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
  return response.json();
}

Promise.all([
  loadJson("release-manifest.json"), loadJson("production-signal.json"),
  loadJson("simulated-account-state.json"),
]).then(([manifest, signal, account]) => {
  renderSignal(signal); renderAccount(account); renderProvenance(manifest, signal, account);
}).catch((error) => {
  const badge = byId("state-badge");
  badge.textContent = "数据不可用"; badge.className = "badge hold";
  byId("reason-codes").textContent = `加载失败：${error.message}`;
});
