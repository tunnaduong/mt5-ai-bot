"""Lớp kiểm soát rủi ro BẮT BUỘC trước mọi lệnh mở mới.

Nguyên tắc: AI chỉ *đề xuất*; RiskManager mới là nơi quyết định có cho vào lệnh hay không,
với khối lượng bao nhiêu. Mọi lý do chặn đều được gom lại để log/debug.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Awaitable, Callable

from ai_client import Action, AIDecision
from models import (
    TRADE_MODE_CLOSEONLY,
    TRADE_MODE_DISABLED,
    TRADE_MODE_LONGONLY,
    TRADE_MODE_SHORTONLY,
    AccountState,
    Position,
    Side,
    SymbolSpec,
    Tick,
)

log = logging.getLogger(__name__)

# (side, entry, sl) -> số tiền lỗ (account currency, số dương) cho 1.0 lot, None nếu không tính được
LossPerLotFn = Callable[[Side, float, float], Awaitable[float | None]]
# (side, volume, price) -> margin cần thiết, None nếu không tính được
MarginFn = Callable[[Side, float, float], Awaitable[float | None]]


@dataclass(frozen=True)
class RiskConfig:
    risk_per_trade_pct: float = 0.5
    max_open_positions: int = 3
    max_positions_per_symbol: int = 1
    daily_max_loss_pct: float = 3.0
    max_spread_points: float = 30.0
    spread_overrides: dict[str, float] = field(default_factory=dict)
    max_spread_atr_ratio: float = 0.25
    min_confidence: float = 0.65
    min_rr: float = 1.2
    max_sl_atr_mult: float = 5.0
    min_sl_atr_mult: float = 0.3
    sl_fallback_atr_mult: float = 0.0
    max_tick_age_sec: int = 180
    max_margin_usage_pct: float = 30.0
    max_lot_cap: float = 0.0

    def max_spread_for(self, symbol: str) -> float:
        return self.spread_overrides.get(symbol, self.max_spread_points)


@dataclass
class TradeProposal:
    """Kết quả risk check. `approved=True` nghĩa là đủ điều kiện gửi lệnh."""

    symbol: str
    action: str
    approved: bool = False
    side: Side | None = None
    volume: float = 0.0
    entry: float | None = None
    sl: float | None = None
    tp: float | None = None
    risk_amount: float = 0.0
    rr: float | None = None
    spread_points: float = 0.0
    reasons: list[str] = field(default_factory=list)

    def block(self, reason: str) -> None:
        self.reasons.append(reason)


def _decimals(step: float) -> int:
    d = Decimal(str(step)).normalize()
    return max(0, -d.as_tuple().exponent)  # type: ignore[operator]


def normalize_volume(spec: SymbolSpec, volume: float) -> float:
    """Làm tròn XUỐNG theo volume_step và kẹp ở volume_max. Trả 0 nếu < volume_min."""
    step = spec.volume_step if spec.volume_step > 0 else 0.01
    steps = math.floor(volume / step + 1e-9)
    vol = round(steps * step, _decimals(step))
    vol = min(vol, spec.volume_max)
    return vol if vol >= spec.volume_min - 1e-12 else 0.0


def normalize_price(spec: SymbolSpec, price: float) -> float:
    """Làm tròn giá theo tick size và số chữ số thập phân của symbol."""
    tick = spec.trade_tick_size if spec.trade_tick_size > 0 else spec.point
    if tick > 0:
        price = round(price / tick) * tick
    return round(price, spec.digits)


def daily_pnl_pct(account: AccountState, day_start_balance: float | None) -> float:
    """% thay đổi equity so với balance đầu ngày (bao gồm cả lãi/lỗ thả nổi)."""
    if not day_start_balance or day_start_balance <= 0:
        return 0.0
    return (account.equity - day_start_balance) / day_start_balance * 100.0


class RiskManager:
    def __init__(self, config: RiskConfig) -> None:
        self.cfg = config

    def daily_loss_hit(self, account: AccountState, day_start_balance: float | None) -> bool:
        return daily_pnl_pct(account, day_start_balance) <= -abs(self.cfg.daily_max_loss_pct)

    def tick_value_loss_per_lot(self, spec: SymbolSpec, entry: float, sl: float) -> float | None:
        """Fallback tính lỗ / 1 lot bằng tick_value (khi order_calc_profit không khả dụng)."""
        if spec.trade_tick_size <= 0 or spec.trade_tick_value <= 0:
            return None
        return abs(entry - sl) / spec.trade_tick_size * spec.trade_tick_value

    async def evaluate(
        self,
        decision: AIDecision,
        spec: SymbolSpec,
        tick: Tick,
        account: AccountState,
        positions: list[Position],
        day_start_balance: float | None,
        atr: float | None,
        tick_age_sec: float = 0.0,
        loss_per_lot_fn: LossPerLotFn | None = None,
        margin_fn: MarginFn | None = None,
    ) -> TradeProposal:
        """Kiểm tra toàn bộ điều kiện và tính khối lượng. Không gửi lệnh."""
        cfg = self.cfg
        p = TradeProposal(symbol=spec.name, action=decision.action.value)
        p.spread_points = tick.spread_points(spec.point)

        if decision.action not in (Action.BUY, Action.SELL):
            p.block(f"action {decision.action.value} không phải lệnh mở")
            return p

        side = Side(decision.action.value)
        p.side = side

        # 1. Confidence
        if decision.confidence < cfg.min_confidence:
            p.block(f"confidence {decision.confidence:.2f} < {cfg.min_confidence:.2f}")

        # 2. Trade mode của symbol
        if spec.trade_mode in (TRADE_MODE_DISABLED, TRADE_MODE_CLOSEONLY):
            p.block(f"symbol trade_mode={spec.trade_mode} (disabled/close-only)")
        elif spec.trade_mode == TRADE_MODE_LONGONLY and side is Side.SELL:
            p.block("symbol chỉ cho phép BUY (long-only)")
        elif spec.trade_mode == TRADE_MODE_SHORTONLY and side is Side.BUY:
            p.block("symbol chỉ cho phép SELL (short-only)")
        if not account.trade_allowed:
            p.block("tài khoản không cho phép giao dịch (trade_allowed=False)")

        # 3. Giá còn "sống"?
        if tick.bid <= 0 or tick.ask <= 0:
            p.block("tick không hợp lệ (bid/ask = 0)")
            return p
        if tick_age_sec > cfg.max_tick_age_sec:
            p.block(f"tick cũ {tick_age_sec:.0f}s > {cfg.max_tick_age_sec}s (thị trường đóng?)")

        # 4. Spread
        max_spread = cfg.max_spread_for(spec.name)
        if p.spread_points > max_spread:
            p.block(f"spread {p.spread_points:.0f} pts > max {max_spread:.0f} pts")
        spread_price = tick.ask - tick.bid
        if atr and cfg.max_spread_atr_ratio > 0 and spread_price > atr * cfg.max_spread_atr_ratio:
            p.block(f"spread {spread_price:.{spec.digits}f} > {cfg.max_spread_atr_ratio:.2f} x ATR {atr:.{spec.digits}f}")

        # 5. Daily loss
        pnl_pct = daily_pnl_pct(account, day_start_balance)
        if pnl_pct <= -abs(cfg.daily_max_loss_pct):
            p.block(f"chạm daily loss limit ({pnl_pct:.2f}% <= -{cfg.daily_max_loss_pct}%)")

        # 6. Số lệnh
        if len(positions) >= cfg.max_open_positions:
            p.block(f"đã có {len(positions)} lệnh mở (max {cfg.max_open_positions})")
        same_symbol = [x for x in positions if x.symbol == spec.name]
        if len(same_symbol) >= cfg.max_positions_per_symbol:
            p.block(f"{spec.name} đã có {len(same_symbol)} lệnh (max {cfg.max_positions_per_symbol}/symbol)")

        # 7. Stop loss
        entry = tick.ask if side is Side.BUY else tick.bid
        p.entry = entry
        sl = decision.stop_loss
        if sl is None and cfg.sl_fallback_atr_mult > 0 and atr:
            sl = entry - atr * cfg.sl_fallback_atr_mult if side is Side.BUY else entry + atr * cfg.sl_fallback_atr_mult
            log.info("%s: AI không gửi SL, dùng fallback %.2f x ATR", spec.name, cfg.sl_fallback_atr_mult)
        if sl is None:
            p.block("thiếu stop loss (bắt buộc)")
            return p
        sl = normalize_price(spec, sl)
        p.sl = sl
        min_dist = spec.min_stop_distance
        if side is Side.BUY:
            if sl >= tick.bid:
                p.block(f"SL {sl} phải < bid {tick.bid} cho lệnh BUY")
                return p
            if tick.bid - sl < min_dist:
                p.block(f"SL quá gần: {tick.bid - sl:.{spec.digits}f} < stops_level {min_dist:.{spec.digits}f}")
        else:
            if sl <= tick.ask:
                p.block(f"SL {sl} phải > ask {tick.ask} cho lệnh SELL")
                return p
            if sl - tick.ask < min_dist:
                p.block(f"SL quá gần: {sl - tick.ask:.{spec.digits}f} < stops_level {min_dist:.{spec.digits}f}")
        sl_dist = abs(entry - sl)
        if atr:
            if cfg.max_sl_atr_mult > 0 and sl_dist > atr * cfg.max_sl_atr_mult:
                p.block(f"SL quá xa: {sl_dist / atr:.1f} x ATR > {cfg.max_sl_atr_mult}")
            if cfg.min_sl_atr_mult > 0 and sl_dist < atr * cfg.min_sl_atr_mult:
                p.block(f"SL quá sát (nhiễu): {sl_dist / atr:.2f} x ATR < {cfg.min_sl_atr_mult}")

        # 8. Take profit & R:R
        tp = decision.take_profit
        if tp is not None:
            tp = normalize_price(spec, tp)
            p.tp = tp
            valid_side = tp > tick.ask if side is Side.BUY else tp < tick.bid
            if not valid_side:
                p.block(f"TP {tp} sai phía cho lệnh {side.value}")
            else:
                ref = tick.bid if side is Side.BUY else tick.ask
                if abs(tp - ref) < min_dist:
                    p.block("TP quá gần (stops_level)")
                p.rr = round(abs(tp - entry) / sl_dist, 2) if sl_dist > 0 else None
                if cfg.min_rr > 0 and (p.rr is None or p.rr < cfg.min_rr):
                    p.block(f"R:R {p.rr} < min {cfg.min_rr}")
        elif cfg.min_rr > 0:
            p.block("thiếu take profit (cần để kiểm tra R:R)")

        # 9. Khối lượng theo % rủi ro
        risk_amount = account.equity * cfg.risk_per_trade_pct / 100.0
        loss_per_lot: float | None = None
        if loss_per_lot_fn is not None:
            try:
                loss_per_lot = await loss_per_lot_fn(side, entry, sl)
            except Exception as exc:  # noqa: BLE001 - fallback an toàn
                log.warning("loss_per_lot_fn lỗi: %s – dùng tick_value", exc)
        if not loss_per_lot or loss_per_lot <= 0:
            loss_per_lot = self.tick_value_loss_per_lot(spec, entry, sl)
        if not loss_per_lot or loss_per_lot <= 0:
            p.block("không tính được giá trị lỗ/lot (tick_value=0?)")
            return p

        raw_volume = risk_amount / loss_per_lot
        if cfg.max_lot_cap > 0:
            raw_volume = min(raw_volume, cfg.max_lot_cap)
        volume = normalize_volume(spec, raw_volume)
        if volume <= 0:
            p.block(
                f"lot tính được {raw_volume:.4f} < volume_min {spec.volume_min} "
                f"(rủi ro min lot = {spec.volume_min * loss_per_lot:.2f} {account.currency} > {risk_amount:.2f})"
            )
            return p
        p.volume = volume
        p.risk_amount = round(volume * loss_per_lot, 2)

        # 10. Margin
        if margin_fn is not None:
            try:
                margin = await margin_fn(side, volume, entry)
            except Exception as exc:  # noqa: BLE001
                margin = None
                log.warning("margin_fn lỗi: %s", exc)
            if margin is None:
                p.block("không tính được margin yêu cầu")
            elif margin > account.margin_free * cfg.max_margin_usage_pct / 100.0:
                p.block(
                    f"margin {margin:.2f} > {cfg.max_margin_usage_pct}% free margin ({account.margin_free:.2f})"
                )

        p.approved = not p.reasons
        return p
