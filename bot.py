"""Điểm vào chính: phối hợp MT5 ↔ AI ↔ Risk Manager ↔ Telegram trên một event loop asyncio.

Chạy:
    python bot.py                 # chạy liên tục (mặc định DRY_RUN=true trong .env)
    python bot.py --once          # chạy 1 nhịp phân tích cho tất cả symbol rồi thoát
    python bot.py --dry-run       # ép chế độ dry-run bất kể .env
    python bot.py --symbols GOLD,EURUSD --no-telegram --log-level DEBUG
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import timedelta
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Awaitable, Callable

import pandas as pd

from ai_client import Action, AIClient, AIResult, build_system_prompt
from config import Settings, get_settings
from indicators import latest_atr, summarize
from models import AccountState, Position, Side, SymbolSpec, Tick
from mt5_client import MT5Client, MT5Error
from risk_manager import RiskConfig, RiskManager, TradeProposal, daily_pnl_pct
from storage import Storage, utc_now
from telegram_bot import NullNotifier, TelegramNotifier, esc

log = logging.getLogger("bot")


@dataclass
class SymbolContext:
    """Toàn bộ dữ liệu cần cho 1 lần ra quyết định trên 1 symbol."""

    symbol: str
    spec: SymbolSpec
    tick: Tick
    account: AccountState
    positions: list[Position]  # tất cả vị thế trên tài khoản
    bot_positions: list[Position]  # vị thế của bot (magic) trên symbol này
    atr: float | None
    tick_age: float
    day_start_balance: float
    snapshot: dict[str, Any] = field(default_factory=dict)


class TradingBot:
    """Vòng lặp trading + giám sát. Đồng thời là `BotController` cho Telegram."""

    def __init__(
        self,
        settings: Settings,
        mt5c: MT5Client,
        ai: AIClient,
        risk: RiskManager,
        storage: Storage,
        notifier: TelegramNotifier | NullNotifier,
    ) -> None:
        self.s, self.mt5, self.ai, self.risk, self.db, self.tg = settings, mt5c, ai, risk, storage, notifier
        self._paused = False
        self._last_bar: dict[str, int] = {}
        self._tick_seen: dict[str, tuple[int, float]] = {}
        self._known_positions: dict[int, Position] = {}
        self._positions_ready = False
        self._pending_closed: dict[int, tuple[Position, int]] = {}
        self._mt5_down_alerted = False
        self._ai_down_alerted = False
        self._loss_alert_day: str | None = None
        self._trade_lock = asyncio.Lock()
        self.started_at = time.time()

    # ================================================================== controller API
    @property
    def is_paused(self) -> bool:
        return self._paused

    @property
    def mode(self) -> str:
        return "DRY-RUN 🧪" if self.mt5.dry_run else "LIVE 🔴"

    def pause(self) -> None:
        self._paused = True
        self.db.log_event("INFO", "control", "paused")
        log.info("Bot PAUSED")

    def resume(self) -> None:
        self._paused = False
        self.db.log_event("INFO", "control", "resumed")
        log.info("Bot RESUMED")

    @staticmethod
    def today() -> str:
        return utc_now().date().isoformat()

    async def day_start_balance(self, account: AccountState) -> float:
        key = f"day_start_balance:{self.today()}"
        value = self.db.get_kv(key)
        if value is None:
            self.db.set_kv(key, str(account.balance))
            return account.balance
        return float(value)

    async def status_text(self) -> str:
        if not self.mt5.connected:
            return "❌ MT5 chưa kết nối."
        acc = await self.mt5.account()
        dsb = await self.day_start_balance(acc)
        positions = await self.mt5.positions()
        stats = self.db.daily_stats(self.today())
        pnl_today = acc.equity - dsb
        uptime_h = (time.time() - self.started_at) / 3600
        return (
            f"<b>📊 STATUS</b>\n"
            f"Trạng thái: {'⏸ TẠM DỪNG' if self._paused else '▶️ ĐANG CHẠY'} | {self.mode}\n"
            f"Tài khoản: {acc.login} ({'REAL' if acc.is_real else 'DEMO'}) – {esc(acc.server)}\n"
            f"Balance: <b>{acc.balance:,.2f} {acc.currency}</b>\n"
            f"Equity: <b>{acc.equity:,.2f}</b> | Free margin: {acc.margin_free:,.2f}\n"
            f"Margin level: {acc.margin_level:,.1f}%\n"
            f"P&amp;L hôm nay (gồm thả nổi): <b>{pnl_today:+,.2f}</b> ({daily_pnl_pct(acc, dsb):+.2f}%)\n"
            f"Lệnh đã đóng hôm nay: {stats['trades']} | Realized: {stats['pnl']:+,.2f}\n"
            f"Vị thế mở: {len(positions)}\n"
            f"Symbols: {esc(', '.join(self.s.symbol_list))} | TF: {self.s.timeframe}\n"
            f"AI lỗi liên tiếp: {self.ai.consecutive_failures} | Uptime: {uptime_h:.1f}h"
        )

    async def positions_text(self) -> str:
        positions = await self.mt5.positions()
        if not positions:
            return "📭 Không có vị thế nào đang mở."
        lines = [f"<b>📂 {len(positions)} vị thế đang mở</b>"]
        total = 0.0
        for p in positions:
            net = p.profit + p.swap
            total += net
            icon = "🟢" if net >= 0 else "🔴"
            owner = "bot" if p.magic == self.mt5.magic else "manual"
            lines.append(
                f"{icon} <b>{esc(p.symbol)}</b> {p.side.value} {p.volume} @ {p.price_open} → {p.price_current}\n"
                f"   SL {p.sl or '-'} | TP {p.tp or '-'} | P&amp;L <b>{net:+,.2f}</b> | #{p.ticket} ({owner})"
            )
        lines.append(f"\nTổng P&amp;L thả nổi: <b>{total:+,.2f}</b>")
        return "\n".join(lines)

    async def stats_text(self, days: int | None = 30) -> str:
        """Thống kê hiệu suất tích luỹ (lệnh đã đóng) trong `days` ngày gần nhất; None = toàn bộ."""
        day_from = (utc_now().date() - timedelta(days=days - 1)).isoformat() if days else None
        st = self.db.performance(day_from)
        scope = f"{days} ngày gần nhất" if days else "toàn bộ"
        if st["trades"] == 0:
            hint = " (DRY-RUN không tạo lệnh thật nên chưa có dữ liệu)" if self.mt5.dry_run else ""
            return f"📈 <b>STATS ({scope})</b>\nChưa có lệnh nào đóng{hint}."
        pf = st["profit_factor"]
        lines = [
            f"<b>📈 STATS ({scope}, từ {st['first_day']})</b>",
            f"Lệnh: {st['trades']} (✅ {st['wins']} / ❌ {st['losses']}) | Winrate <b>{st['winrate']:.1f}%</b>",
            f"P&amp;L realized: <b>{st['pnl']:+,.2f}</b>",
            f"TB thắng {st['avg_win']:+.2f} | TB thua {st['avg_loss']:+.2f} | "
            f"Profit factor {'∞' if pf == float('inf') else f'{pf:.2f}'}",
            f"Max drawdown {st['max_drawdown']:.2f} | Chuỗi thua dài nhất {st['max_losing_streak']}",
            "",
            "<b>Theo mã:</b>",
        ]
        for sym, x in st["by_symbol"].items():
            lines.append(f"• {esc(sym)}: {x['trades']} lệnh, WR {x['winrate']:.0f}%, P&amp;L {x['pnl']:+,.2f}")
        lines.append("\n<b>7 ngày gần nhất:</b>")
        for day, x in list(st["by_day"].items())[-7:]:
            lines.append(f"• {day}: {x['trades']} lệnh, WR {x['winrate']:.0f}%, {x['pnl']:+,.2f}")
        return "\n".join(lines)

    async def close_all_positions(self) -> str:
        async with self._trade_lock:
            positions = await self.mt5.positions()
            if not positions:
                return "📭 Không có vị thế nào để đóng."
            ok, fail = [], []
            for p in positions:
                res = await self.mt5.close_position(p, comment="close_all telegram")
                self.db.log_order(p.symbol, f"CLOSE_{p.side.value}", p.volume, res.price, None, None,
                                  res.dry_run, res.success, res.retcode, res.comment, p.ticket, res.request)
                (ok if res.success else fail).append(f"#{p.ticket} {esc(p.symbol)}: {res.retcode} {esc(res.comment)}")
        self.db.log_event("WARNING", "close_all", f"ok={len(ok)} fail={len(fail)}")
        prefix = "🧪 (DRY-RUN – chỉ order_check, KHÔNG đóng thật)\n" if self.mt5.dry_run else ""
        text = f"{prefix}<b>🧹 Close all:</b> ✅ {len(ok)} | ❌ {len(fail)}"
        if fail:
            text += "\n" + "\n".join(fail)
        return text

    async def analyze_text(self, symbol: str) -> str:
        """/analyze: chạy AI ngay, chạy risk check thử, KHÔNG gửi lệnh."""
        try:
            ctx = await self.build_context(symbol)
        except MT5Error as exc:
            return f"❌ {esc(exc)}"
        result = await self._ask_ai(ctx)
        d = result.decision
        proposal = await self._evaluate(ctx, result) if d.action in (Action.BUY, Action.SELL) else None
        main = ctx.snapshot["timeframes"].get(self.s.timeframe, {})
        lines = [
            f"<b>🔎 {esc(symbol)}</b> ({self.s.timeframe}) bid {ctx.tick.bid} / ask {ctx.tick.ask} "
            f"| spread {ctx.tick.spread_points(ctx.spec.point):.0f} pts",
            f"EMA20 {main.get('ema20')} | EMA50 {main.get('ema50')} | EMA200 {main.get('ema200')}",
            f"RSI14 {main.get('rsi14')} | ATR14 {main.get('atr14')} | Trend {main.get('trend')} "
            f"| Nến: {main.get('last_candle')}",
        ]
        for tf, summ in ctx.snapshot["timeframes"].items():
            if tf != self.s.timeframe and "trend" in summ:
                lines.append(f"{tf}: trend {summ['trend']}, RSI {summ['rsi14']}")
        lines += [
            "",
            f"🤖 AI: <b>{d.action.value}</b> (conf {d.confidence:.2f})"
            + (" ⚠️ fallback" if result.is_fallback else ""),
            f"SL {d.stop_loss or '-'} | TP {d.take_profit or '-'}",
            f"💬 {esc(d.reasoning)}",
        ]
        if proposal:
            if proposal.approved:
                lines.append(f"🛡 Risk: ✅ sẽ PASS – lot {proposal.volume}, rủi ro {proposal.risk_amount} "
                             f"{ctx.account.currency} ({proposal.risk_pct}%), R:R {proposal.rr}")
                lines += [f"ℹ️ {esc(n)}" for n in proposal.notes]
            else:
                lines.append("🛡 Risk: ⛔ BLOCK – " + esc("; ".join(proposal.reasons)))
        lines.append("<i>(Chỉ phân tích, không vào lệnh)</i>")
        return "\n".join(lines)

    # ================================================================== context & AI
    def _tick_age(self, tick: Tick) -> float:
        """Tuổi của tick đo bằng đồng hồ local (tránh lệch múi giờ server broker)."""
        now = time.monotonic()
        prev = self._tick_seen.get(tick.symbol)
        if prev is None or prev[0] != tick.time:
            self._tick_seen[tick.symbol] = (tick.time, now)
            return 0.0
        return now - prev[1]

    async def build_context(self, symbol: str, main_df: pd.DataFrame | None = None) -> SymbolContext:
        spec = await self.mt5.symbol_spec(symbol)
        tick = await self.mt5.tick(symbol)
        account = await self.mt5.account()
        positions = await self.mt5.positions()
        if main_df is None:
            main_df = await self.mt5.bars(symbol, self.s.timeframe, self.s.bars_count)
        timeframes = {self.s.timeframe: summarize(main_df, self.s.timeframe, spec.digits)}
        for tf in self.s.context_timeframe_list:
            if tf == self.s.timeframe:
                continue
            try:
                timeframes[tf] = summarize(await self.mt5.bars(symbol, tf, self.s.bars_count), tf, spec.digits, 3)
            except MT5Error as exc:
                timeframes[tf] = {"timeframe": tf, "error": str(exc)[:100]}
        atr = latest_atr(main_df)
        dsb = await self.day_start_balance(account)
        bot_positions = [p for p in positions if p.symbol == symbol and p.magic == self.mt5.magic]
        ctx = SymbolContext(symbol, spec, tick, account, positions, bot_positions, atr, self._tick_age(tick), dsb)
        ctx.snapshot = {
            "symbol": symbol,
            "description": spec.description,
            "digits": spec.digits,
            "price": {"bid": tick.bid, "ask": tick.ask, "spread_points": tick.spread_points(spec.point)},
            "broker_min_stop_distance": round(spec.min_stop_distance, spec.digits),
            "main_timeframe": self.s.timeframe,
            "timeframes": timeframes,
            "open_positions": [
                {"ticket": p.ticket, "side": p.side.value, "volume": p.volume, "price_open": p.price_open,
                 "sl": p.sl, "tp": p.tp, "profit": round(p.profit, 2)}
                for p in bot_positions
            ],
            "account": {
                "currency": account.currency, "balance": round(account.balance, 2),
                "equity": round(account.equity, 2), "margin_level": round(account.margin_level, 1),
                "daily_pnl_pct": round(daily_pnl_pct(account, dsb), 2), "open_positions_total": len(positions),
            },
            "risk_rules": {
                "min_rr": self.s.min_rr,
                "max_spread_points": self.risk.cfg.max_spread_for(symbol),
                "max_risk_pct_per_trade": self.risk.max_risk_pct,
            },
        }
        affordable = self.risk.max_affordable_sl_distance(spec, account.equity)
        if affordable is not None and atr and affordable < atr * max(self.s.max_sl_atr_mult, 1.0):
            # Chỉ gửi khi đây thực sự là ràng buộc (tài khoản nhỏ) để không làm nhiễu AI
            ctx.snapshot["risk_rules"]["max_affordable_sl_distance"] = round(affordable, spec.digits)
        return ctx

    async def _ask_ai(self, ctx: SymbolContext) -> AIResult:
        result = await self.ai.decide(ctx.snapshot)
        d = result.decision
        self.db.log_ai_call(ctx.symbol, result.ok, result.attempts, result.latency_ms, ctx.snapshot,
                            result.raw_text, d.model_dump(mode="json"), result.error)
        log.info("AI %s -> %s conf=%.2f sl=%s tp=%s (%dms, ok=%s) | %s", ctx.symbol, d.action.value,
                 d.confidence, d.stop_loss, d.take_profit, result.latency_ms, result.ok, d.reasoning[:120])
        await self._check_ai_health()
        return result

    async def _evaluate(self, ctx: SymbolContext, result: AIResult) -> TradeProposal:
        sym = ctx.symbol
        loss_fn: Callable[[Side, float, float], Awaitable[float | None]] = (
            lambda side, entry, sl: self.mt5.loss_per_lot(sym, side, entry, sl)
        )
        margin_fn: Callable[[Side, float, float], Awaitable[float | None]] = (
            lambda side, vol, price: self.mt5.margin_required(sym, side, vol, price)
        )
        p = await self.risk.evaluate(
            result.decision, ctx.spec, ctx.tick, ctx.account, ctx.positions, ctx.day_start_balance,
            ctx.atr, ctx.tick_age, loss_per_lot_fn=loss_fn, margin_fn=margin_fn,
        )
        self.db.log_risk(sym, p.action, result.decision.confidence, p.approved, p.volume, p.entry, p.sl, p.tp,
                         p.risk_amount, p.rr, p.reasons)
        log.info("RISK %s %s: %s %s", sym, p.action, "APPROVED" if p.approved else "BLOCKED",
                 f"lot={p.volume} risk={p.risk_amount} rr={p.rr}" if p.approved else "; ".join(p.reasons))
        return p

    # ================================================================== trading
    async def trading_once(self, force: bool = False) -> None:
        """1 nhịp: với mỗi symbol, nếu có nến mới (hoặc force) thì hỏi AI và xử lý."""
        if self._paused or not self.mt5.connected:
            return
        for symbol in self.s.symbol_list:
            if self._paused:
                return
            try:
                await self._process_symbol(symbol, force)
            except MT5Error as exc:
                log.error("%s: lỗi MT5: %s", symbol, exc)
                self.db.log_event("ERROR", "mt5", f"{symbol}: {exc}")
            except Exception as exc:  # noqa: BLE001 - 1 symbol lỗi không được làm sập bot
                log.exception("%s: lỗi không mong đợi", symbol)
                self.db.log_event("ERROR", "unexpected", f"{symbol}: {exc!r}")

    async def _process_symbol(self, symbol: str, force: bool) -> None:
        main_df = await self.mt5.bars(symbol, self.s.timeframe, self.s.bars_count)
        bar_time = int(main_df["time"].iloc[-1])
        if not force and self._last_bar.get(symbol) == bar_time:
            return
        self._last_bar[symbol] = bar_time
        ctx = await self.build_context(symbol, main_df)
        result = await self._ask_ai(ctx)
        await self._act(ctx, result)

    async def _act(self, ctx: SymbolContext, result: AIResult) -> None:
        d = result.decision
        if d.action is Action.HOLD:
            return
        if d.action is Action.CLOSE:
            await self._close_symbol_positions(ctx, d.reasoning)
            return

        await self._refresh_market(ctx)
        proposal = await self._evaluate(ctx, result)
        if not proposal.approved:
            if d.confidence >= self.s.notify_min_confidence:
                self.tg.notify(
                    f"🛡 <b>Risk BLOCK</b> {esc(ctx.symbol)} {d.action.value} (conf {d.confidence:.2f})\n"
                    f"• " + "\n• ".join(esc(r) for r in proposal.reasons)
                )
            return

        async with self._trade_lock:
            if self._paused:
                log.info("%s: bị pause trước khi gửi lệnh – bỏ qua", ctx.symbol)
                return
            assert proposal.side and proposal.sl
            res = await self.mt5.open_position(ctx.spec, proposal.side, proposal.volume, proposal.sl, proposal.tp,
                                               comment=f"ai c{d.confidence:.2f}")
        self.db.log_order(ctx.symbol, proposal.side.value, proposal.volume, res.price, proposal.sl, proposal.tp,
                          res.dry_run, res.success, res.retcode, res.comment, res.ticket, res.request)
        if res.success:
            tag = "🧪 DRY-RUN (order_check OK, không gửi lệnh)" if res.dry_run else "✅ ĐÃ VÀO LỆNH"
            self.tg.notify(
                f"{tag}\n<b>{esc(ctx.symbol)} {proposal.side.value}</b> {proposal.volume} lot @ {res.price}\n"
                f"SL {proposal.sl} | TP {proposal.tp or '-'} | R:R {proposal.rr}\n"
                f"Rủi ro: {proposal.risk_amount} {ctx.account.currency} ({proposal.risk_pct}%) "
                f"| conf {d.confidence:.2f}\n💬 {esc(d.reasoning[:300])}"
                + "".join(f"\nℹ️ {esc(n)}" for n in proposal.notes)
                + (f"\nTicket #{res.ticket}" if res.ticket else "")
            )
        else:
            self.tg.notify(f"❌ <b>Lệnh {esc(ctx.symbol)} {proposal.side.value} thất bại</b>\n"
                           f"retcode {res.retcode}: {esc(res.comment)}")
            self.db.log_event("ERROR", "order", f"{ctx.symbol} retcode={res.retcode} {res.comment}")

    async def _refresh_market(self, ctx: SymbolContext) -> None:
        """Cập nhật lại giá/tài khoản/vị thế sau khi chờ AI (có thể mất hàng chục giây)."""
        ctx.tick = await self.mt5.tick(ctx.symbol)
        ctx.tick_age = self._tick_age(ctx.tick)
        ctx.account = await self.mt5.account()
        ctx.positions = await self.mt5.positions()

    async def _close_symbol_positions(self, ctx: SymbolContext, reason: str) -> None:
        if not ctx.bot_positions:
            log.info("%s: AI muốn CLOSE nhưng bot không có vị thế", ctx.symbol)
            return
        async with self._trade_lock:
            for p in ctx.bot_positions:
                res = await self.mt5.close_position(p, comment="ai close")
                self.db.log_order(p.symbol, f"CLOSE_{p.side.value}", p.volume, res.price, None, None,
                                  res.dry_run, res.success, res.retcode, res.comment, p.ticket, res.request)
                status = ("🧪 DRY-RUN close" if res.dry_run else "🔻 AI đóng lệnh") if res.success else "❌ Đóng lỗi"
                self.tg.notify(f"{status} #{p.ticket} {esc(p.symbol)} – P&amp;L hiện tại {p.profit:+.2f}\n"
                               f"💬 {esc(reason[:300])}" + ("" if res.success else f"\n{res.retcode} {esc(res.comment)}"))

    # ================================================================== monitoring
    async def monitor_once(self) -> None:
        """Kiểm tra kết nối, theo dõi lệnh đóng (SL/TP), daily loss, sức khoẻ AI, báo cáo cuối ngày."""
        if not await self.mt5.ensure_connected():
            if not self._mt5_down_alerted:
                self._mt5_down_alerted = True
                self.db.log_event("CRITICAL", "mt5", "connection lost")
                self.tg.notify("🚨 <b>MẤT KẾT NỐI MT5</b> – đang thử kết nối lại. Bot không thể giao dịch/giám sát!")
            return
        if self._mt5_down_alerted:
            self._mt5_down_alerted = False
            self.tg.notify("✅ Đã kết nối lại MT5.")

        account = await self.mt5.account()
        dsb = await self.day_start_balance(account)
        await self._track_positions(await self.mt5.positions())
        for symbol in self.s.symbol_list:  # cập nhật tuổi tick
            try:
                self._tick_age(await self.mt5.tick(symbol))
            except MT5Error:
                pass
        await self._check_daily_loss(account, dsb)
        await self._maybe_daily_summary(account)

    async def _track_positions(self, positions: list[Position]) -> None:
        current = {p.ticket: p for p in positions}
        if not self._positions_ready:
            self._known_positions, self._positions_ready = current, True
            return
        for ticket, pos in self._known_positions.items():
            if ticket not in current:
                self._pending_closed[ticket] = (pos, 0)
        for ticket, pos in current.items():
            if ticket not in self._known_positions and pos.magic != self.mt5.magic:
                self.tg.notify(f"ℹ️ Phát hiện vị thế mới (manual) #{ticket} {esc(pos.symbol)} {pos.side.value} {pos.volume}")
        self._known_positions = current

        for ticket, (pos, tries) in list(self._pending_closed.items()):
            info = await self.mt5.closed_trade_info(pos)
            if info is None and tries < 10:
                self._pending_closed[ticket] = (pos, tries + 1)  # lịch sử chưa đồng bộ, thử lại sau
                continue
            del self._pending_closed[ticket]
            if info is None:
                self.tg.notify(f"ℹ️ Vị thế #{ticket} {esc(pos.symbol)} đã đóng (không đọc được lịch sử deal). "
                               f"P&amp;L cuối ≈ {pos.profit:+.2f}")
                continue
            self.db.log_close(info)
            icon = {"Stop Loss": "🛑", "Take Profit": "🎯"}.get(info.reason, "🔒")
            self.tg.notify(
                f"{icon} <b>ĐÓNG LỆNH</b> {esc(info.symbol)} {info.side} {info.volume} – lý do: <b>{esc(info.reason)}</b>\n"
                f"{info.open_price} → {info.close_price}\n"
                f"P&amp;L: <b>{info.profit:+,.2f}</b> ({info.balance_change_pct:+.2f}% balance)\n"
                f"Balance: {info.balance_after:,.2f}"
            )

    async def _check_daily_loss(self, account: AccountState, dsb: float) -> None:
        today = self.today()
        if self._loss_alert_day == today or not self.risk.daily_loss_hit(account, dsb):
            return
        self._loss_alert_day = today
        pct = daily_pnl_pct(account, dsb)
        self.db.log_event("CRITICAL", "daily_loss", f"{pct:.2f}%")
        msg = (f"🚨 <b>CHẠM DAILY LOSS LIMIT</b> {pct:.2f}% (giới hạn -{self.s.daily_max_loss_pct}%)\n"
               f"Equity {account.equity:,.2f} / đầu ngày {dsb:,.2f}. Bot ngừng mở lệnh mới tới hết ngày (UTC).")
        if self.s.daily_loss_close_all:
            msg += "\n" + await self.close_all_positions()
        self.tg.notify(msg)

    async def _maybe_daily_summary(self, account: AccountState) -> None:
        now = utc_now()
        key = f"summary_sent:{now.date().isoformat()}"
        if now.hour < self.s.daily_summary_hour_utc or self.db.get_kv(key):
            return
        self.db.set_kv(key, "1")
        st = self.db.daily_stats(now.date().isoformat())
        dsb = await self.day_start_balance(account)
        self.tg.notify(
            f"<b>📅 TỔNG KẾT NGÀY {now.date().isoformat()} (UTC)</b>\n"
            f"Số lệnh đóng: {st['trades']} (✅ {st['wins']} / ❌ {st['losses']})\n"
            f"Winrate: {st['winrate']:.1f}%\n"
            f"P&amp;L realized: <b>{st['pnl']:+,.2f} {account.currency}</b> | best {st['best']:+.2f} / worst {st['worst']:+.2f}\n"
            f"Equity: {account.equity:,.2f} ({daily_pnl_pct(account, dsb):+.2f}% so với đầu ngày)\n"
            f"Chế độ: {self.mode}{' | ⏸ đang pause' if self._paused else ''}"
        )

    async def _check_ai_health(self) -> None:
        n = self.ai.consecutive_failures
        if n >= self.s.ai_failure_alert_threshold and not self._ai_down_alerted:
            self._ai_down_alerted = True
            self.db.log_event("ERROR", "ai", f"{n} consecutive failures")
            self.tg.notify(f"🚨 <b>AI endpoint lỗi {n} lần liên tiếp</b> – bot đang fallback HOLD (không vào lệnh).")
        elif n == 0 and self._ai_down_alerted:
            self._ai_down_alerted = False
            self.tg.notify("✅ AI endpoint đã hoạt động lại.")

    # ================================================================== loops
    async def _loop(self, name: str, fn: Callable[[], Awaitable[None]], interval: float, stop: asyncio.Event) -> None:
        while not stop.is_set():
            try:
                await fn()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                log.exception("Loop %s lỗi", name)
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass

    async def run_forever(self, stop: asyncio.Event) -> None:
        await asyncio.gather(
            self._loop("monitor", self.monitor_once, self.s.monitor_interval_sec, stop),
            self._loop("trading", self.trading_once, self.s.loop_interval_sec, stop),
        )


# ====================================================================== bootstrap
def setup_logging(log_dir: str, level: str) -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(level.upper())
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    file_handler = RotatingFileHandler(Path(log_dir) / "bot.log", maxBytes=5_000_000, backupCount=10, encoding="utf-8")
    file_handler.setFormatter(fmt)
    root.handlers[:] = [console, file_handler]
    for noisy in ("httpx", "httpcore", "telegram", "apscheduler"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="MT5 AI trading bot (FxPro) + Telegram")
    ap.add_argument("--once", action="store_true", help="chạy 1 nhịp phân tích cho mọi symbol rồi thoát")
    ap.add_argument("--dry-run", action="store_true", help="ép DRY_RUN=true (không gửi lệnh thật)")
    ap.add_argument("--symbols", help="ghi đè SYMBOLS, ví dụ GOLD,EURUSD")
    ap.add_argument("--no-telegram", action="store_true", help="tắt Telegram")
    ap.add_argument("--log-level", help="DEBUG/INFO/WARNING")
    return ap.parse_args(argv)


def build_risk_config(s: Settings) -> RiskConfig:
    return RiskConfig(
        risk_per_trade_pct=s.risk_per_trade_pct, max_open_positions=s.max_open_positions,
        max_positions_per_symbol=s.max_positions_per_symbol, daily_max_loss_pct=s.daily_max_loss_pct,
        max_spread_points=s.max_spread_points, spread_overrides=s.spread_overrides,
        max_spread_atr_ratio=s.max_spread_atr_ratio, min_confidence=s.min_confidence, min_rr=s.min_rr,
        max_sl_atr_mult=s.max_sl_atr_mult, min_sl_atr_mult=s.min_sl_atr_mult,
        sl_fallback_atr_mult=s.sl_fallback_atr_mult, max_tick_age_sec=s.max_tick_age_sec,
        max_margin_usage_pct=s.max_margin_usage_pct, max_lot_cap=s.max_lot_cap,
        min_lot_max_risk_pct=s.min_lot_max_risk_pct,
    )


async def main_async(args: argparse.Namespace) -> int:
    s = get_settings()
    overrides: dict[str, Any] = {}
    if args.dry_run:
        overrides["dry_run"] = True
    if args.symbols:
        overrides["symbols"] = args.symbols
    if args.log_level:
        overrides["log_level"] = args.log_level
    if overrides:
        s = s.model_copy(update=overrides)
    setup_logging(s.log_dir, s.log_level)
    log.info("Khởi động bot – mode=%s symbols=%s tf=%s", "DRY-RUN" if s.dry_run else "LIVE", s.symbol_list, s.timeframe)

    storage = Storage(s.db_path)
    ai = AIClient(
        url=s.ai_endpoint_url, api_key=s.ai_api_key.get_secret_value(), model=s.ai_model, style=s.ai_api_style,
        timeout=s.ai_timeout_sec, max_retries=s.ai_max_retries, temperature=s.ai_temperature,
        json_mode=s.ai_json_mode,
        fallback_models=[m.strip() for m in s.ai_fallback_models.split(",") if m.strip()],
        system_prompt=build_system_prompt(s.ai_reasoning_language, s.min_rr, s.min_sl_atr_mult, s.max_sl_atr_mult),
    )
    notifier: TelegramNotifier | NullNotifier = (
        TelegramNotifier(s.telegram_bot_token.get_secret_value(), s.telegram_admin_chat_id, s.telegram_min_interval_sec)
        if s.telegram_enabled and not args.no_telegram else NullNotifier()
    )
    mt5c: MT5Client | None = None
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except (NotImplementedError, RuntimeError):  # Windows: Ctrl+C -> KeyboardInterrupt/CancelledError
            pass

    try:
        mt5c = MT5Client(
            login=s.mt5_login, password=s.mt5_password.get_secret_value(), server=s.mt5_server, path=s.mt5_path,
            magic=s.mt5_magic, deviation=s.mt5_deviation, dry_run=s.dry_run, order_retries=s.mt5_order_retries,
        )
        warnings = await mt5c.connect()
        account = await mt5c.account()
        if not s.dry_run and account.is_real and not s.allow_real_account:
            log.critical("Tài khoản REAL nhưng ALLOW_REAL_ACCOUNT=false -> từ chối chạy LIVE. Thoát.")
            return 2
        bot = TradingBot(s, mt5c, ai, RiskManager(build_risk_config(s)), storage, notifier)
        notifier.attach(bot)
        await notifier.start(polling=not args.once)
        notifier.notify(
            f"🚀 <b>Bot khởi động</b> – {bot.mode}\n"
            f"Tài khoản {account.login} ({'REAL' if account.is_real else 'DEMO'}) {esc(account.server)}\n"
            f"Balance {account.balance:,.2f} {account.currency} | Symbols: {esc(', '.join(s.symbol_list))} "
            f"| TF {s.timeframe}\nRisk/lệnh {s.risk_per_trade_pct}% | Daily loss {s.daily_max_loss_pct}%"
            + "".join(f"\n⚠️ {esc(w)}" for w in warnings)
        )
        if args.once:
            await bot.monitor_once()
            await bot.trading_once(force=True)
        else:
            await bot.run_forever(stop)
        return 0
    except MT5Error as exc:
        log.critical("MT5: %s", exc)
        notifier.notify(f"🚨 <b>Bot dừng</b>: {esc(exc)}")
        return 1
    finally:
        log.info("Đang tắt bot...")
        notifier.notify("🛑 Bot đã dừng.")
        await notifier.stop()
        await ai.aclose()
        if mt5c is not None:
            await mt5c.shutdown()
        storage.close()
        log.info("Đã tắt hoàn toàn.")


def main() -> None:
    args = parse_args()
    try:
        sys.exit(asyncio.run(main_async(args)))
    except KeyboardInterrupt:
        print("Đã dừng bởi người dùng (Ctrl+C).")


if __name__ == "__main__":
    main()
