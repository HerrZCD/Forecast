#!/usr/bin/env python3
"""
daily_rebalance.py
==================
每日调仓脚本：
1. 连接 Alpaca，获取账户总权益
2. 从 DuckDB tft_alpha_scores 表获取最新日期 rank_score 前 5 名
3. 等权计算目标仓位
4. 对比当前持仓，计算差额
5. 设最小交易阈值，避免小额交易浪费手续费
6. 提交市价单完成调仓
"""

import os
import sys
import argparse
import logging
from datetime import datetime

import duckdb
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.common.exceptions import APIError
from alpaca.trading.enums import AssetStatus

# ============================================================
# 配置
# ============================================================
DB_PATH = os.environ.get(
    "DUCKDB_PATH",
    os.path.join("/home/zhangchundong/DailySync/data_loader", "stock_prices.duckdb"),
)

TOP_N = 5                      # 等权持有前 N 名
MIN_TRADE_USD = 100.0          # 最小交易金额阈值（美元），低于此金额不交易
MIN_TRADE_PCT = 0.02           # 最小交易比例阈值（占目标仓位），低于此不交易

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def clean_ticker(ticker: str) -> str:
    """AKAM.O -> AKAM, MSFT.O -> MSFT, strip exchange suffix."""
    return ticker.split(".")[0] if "." in ticker else ticker


# ============================================================
# Alpaca 连接
# ============================================================
def get_trading_client(paper: bool = True) -> TradingClient:
    if paper:
        api_key = os.environ.get("ALPACA_PAPER_API_KEY", "PKUZ5OCCZOBMFKI7XR642NKA4B")
        secret_key = os.environ.get("ALPACA_PAPER_SECRET_KEY", "AbLphw87GQygFYkUKtNMCpxBhg1xnrcKDEadcZi78sD4")
        env_hint = "ALPACA_PAPER_API_KEY / ALPACA_PAPER_SECRET_KEY"
    else:
        api_key = os.environ.get("ALPACA_LIVE_API_KEY")
        secret_key = os.environ.get("ALPACA_LIVE_SECRET_KEY")
        env_hint = "ALPACA_LIVE_API_KEY / ALPACA_LIVE_SECRET_KEY"

    if not api_key or not secret_key:
        raise RuntimeError(f"请设置环境变量 {env_hint}")
    return TradingClient(api_key, secret_key, paper=paper)


# ============================================================
# DuckDB: 获取最新交易日 top N 股票
# ============================================================
def get_top_tickers(n: int = TOP_N, target_date: str = None) -> list[dict]:
    """
    返回最新日期 rank_score 前 n 名，格式:
    [{"ticker": "AAPL", "rank_score": 0.98}, ...]
    """
    con = duckdb.connect(DB_PATH, read_only=True)
    try:
        if target_date:
            query = """
                SELECT ticker, rank_score
                FROM tft_alpha_scores
                WHERE date = ?
                ORDER BY rank_score DESC
                LIMIT ?
            """
            rows = con.execute(query, [target_date, n]).fetchall()
            used_date = target_date
        else:
            query = """
                WITH latest AS (
                    SELECT MAX(date) AS d FROM tft_alpha_scores
                )
                SELECT t.ticker, t.rank_score
                FROM tft_alpha_scores t, latest l
                WHERE t.date = l.d
                ORDER BY t.rank_score DESC
                LIMIT ?
            """
            rows = con.execute(query, [n]).fetchall()
            used_date = con.execute(
                "SELECT MAX(date) FROM tft_alpha_scores"
            ).fetchone()[0]

        log.info(f"Score date: {used_date}, top {n} tickers:")
        results = []
        for ticker, score in rows:
            sym = clean_ticker(ticker)
            log.info(f"  {sym:>6s}  rank_score={score:.4f}  (raw: {ticker})")
            results.append({"ticker": sym, "rank_score": score})
        return results
    finally:
        con.close()


# ============================================================
# 核心调仓逻辑
# ============================================================
def rebalance(
    client: TradingClient,
    top_tickers: list[dict],
    min_trade_usd: float = MIN_TRADE_USD,
    min_trade_pct: float = MIN_TRADE_PCT,
    dry_run: bool = False,
):
    # ---------- 1. 账户信息 ----------
    account = client.get_account()
    equity = float(account.equity)
    buying_power = float(account.buying_power)
    log.info(f"Account equity: ${equity:,.2f}, buying power: ${buying_power:,.2f}")

    if equity <= 0:
        log.error("账户权益为 0，无法调仓")
        return

    # ---------- 2. 当前持仓 ----------
    positions = client.get_all_positions()
    current_holdings: dict[str, float] = {}  # ticker -> market_value
    current_qty: dict[str, float] = {}       # ticker -> qty (支持小数股)
    for pos in positions:
        current_holdings[pos.symbol] = float(pos.market_value)
        current_qty[pos.symbol] = float(pos.qty)

    log.info(f"Current positions ({len(current_holdings)}):")
    for sym, val in sorted(current_holdings.items()):
        log.info(f"  {sym:>6s}  ${val:,.2f}  ({current_qty[sym]:.4f} shares)")

    # ---------- 3. 目标仓位 ----------
    target_tickers = [t["ticker"] for t in top_tickers]
    target_weight = 1.0 / len(target_tickers)
    target_value_per_stock = equity * target_weight

    log.info(f"Target: equal-weight {len(target_tickers)} stocks, "
             f"${target_value_per_stock:,.2f} each ({target_weight:.1%})")

    # ---------- 4. 计算调仓差额 ----------
    # 需要卖出的：当前持有但不在目标中的
    sells: list[dict] = []
    for sym, val in current_holdings.items():
        if sym not in target_tickers:
            sells.append({"ticker": sym, "side": "sell", "notional": val,
                          "reason": "not in target"})

    # 需要调整的：在目标中的
    buys: list[dict] = []
    for ticker in target_tickers:
        current_val = current_holdings.get(ticker, 0.0)
        diff = target_value_per_stock - current_val

        # 检查阈值
        abs_diff = abs(diff)
        if abs_diff < min_trade_usd:
            log.info(f"  {ticker}: skip (diff ${diff:+,.2f} < min ${min_trade_usd})")
            continue
        if target_value_per_stock > 0 and (abs_diff / target_value_per_stock) < min_trade_pct:
            log.info(f"  {ticker}: skip (diff {abs_diff/target_value_per_stock:.1%} < min {min_trade_pct:.1%})")
            continue

        if diff > 0:
            buys.append({"ticker": ticker, "side": "buy", "notional": diff})
        elif diff < 0:
            sells.append({"ticker": ticker, "side": "sell", "notional": abs(diff),
                          "reason": "reduce to target"})

    # ---------- 5. 执行卖出（先卖后买，释放资金） ----------
    log.info(f"=== Orders: {len(sells)} sells, {len(buys)} buys ===")

    for order in sells:
        ticker = order["ticker"]
        reason = order.get("reason", "")

        if ticker not in target_tickers:
            # 全部清仓
            qty = current_qty.get(ticker, 0)
            if qty <= 0:
                continue
            log.info(f"  SELL ALL {ticker}: {qty:.4f} shares ({reason})")
            if not dry_run:
                try:
                    client.close_position(ticker)
                except APIError as e:
                    log.error(f"  Failed to close {ticker}: {e}")
        else:
            # 部分卖出（用 notional）
            notional = order["notional"]
            log.info(f"  SELL {ticker}: ${notional:,.2f} ({reason})")
            if not dry_run:
                try:
                    req = MarketOrderRequest(
                        symbol=ticker,
                        notional=round(notional, 2),
                        side=OrderSide.SELL,
                        time_in_force=TimeInForce.DAY,
                    )
                    client.submit_order(req)
                except APIError as e:
                    log.error(f"  Failed to sell {ticker}: {e}")

    # ---------- 6. 执行买入 ----------
    for order in buys:
        ticker = order["ticker"]
        notional = order["notional"]
        log.info(f"  BUY {ticker}: ${notional:,.2f}")
        if not dry_run:
            try:
                req = MarketOrderRequest(
                    symbol=ticker,
                    notional=round(notional, 2),
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                )
                client.submit_order(req)
            except APIError as e:
                log.error(f"  Failed to buy {ticker}: {e}")

    log.info("Rebalance complete.")


# ============================================================
# CLI
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(description="Daily portfolio rebalance via Alpaca")
    p.add_argument("--date", type=str, default=None,
                   help="Score date (YYYY-MM-DD). Default: latest in DB.")
    p.add_argument("--top", type=int, default=TOP_N,
                   help=f"Number of top stocks to hold (default {TOP_N})")
    p.add_argument("--min-trade", type=float, default=MIN_TRADE_USD,
                   help=f"Minimum trade amount in USD (default {MIN_TRADE_USD})")
    p.add_argument("--min-trade-pct", type=float, default=MIN_TRADE_PCT,
                   help=f"Minimum trade as pct of target (default {MIN_TRADE_PCT})")
    p.add_argument("--dry-run", action="store_true",
                   help="Only print orders, do not execute")
    p.add_argument("--live", action="store_true",
                   help="Use live trading (default: paper)")
    return p.parse_args()


def check_market_open(client: TradingClient) -> bool:
    """检查美股市场是否开盘，未开盘则返回 False。"""
    clock = client.get_clock()
    if not clock.is_open:
        log.warning(f"Market is CLOSED. Next open: {clock.next_open}")
        return False
    log.info(f"Market is OPEN. Closes at {clock.next_close}")
    return True


def main():
    args = parse_args()

    log.info("=" * 60)
    log.info("Daily Rebalance Start")
    log.info("=" * 60)

    # 0. 连接 Alpaca 并检查开盘状态
    paper = not args.live
    log.info(f"Trading mode: {'PAPER' if paper else '*** LIVE ***'}")
    client = get_trading_client(paper=paper)

    if not args.dry_run and not check_market_open(client):
        log.info("Market closed, exiting.")
        return 0

    # 1. 获取 top tickers
    top_tickers = get_top_tickers(n=args.top, target_date=args.date)
    if not top_tickers:
        log.error("No tickers found in tft_alpha_scores. Abort.")
        return 1

    # 2. 调仓
    rebalance(
        client,
        top_tickers,
        min_trade_usd=args.min_trade,
        min_trade_pct=args.min_trade_pct,
        dry_run=args.dry_run,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
