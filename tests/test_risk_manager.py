"""Unit test RiskManager – không cần MetaTrader5 (dùng dataclass giả lập + async fake)."""
from __future__ import annotations

import pytest

from ai_client import Action, AIDecision
from models import (
    TRADE_MODE_CLOSEONLY,
    TRADE_MODE_LONGONLY,
    AccountState,
    Position,
    Side,
    SymbolSpec,
    Tick,
)
from risk_manager import RiskConfig, RiskManager, normalize_price, normalize_volume

EURUSD = SymbolSpec(
    name="EURUSD", digits=5, point=0.00001, trade_tick_size=0.00001, trade_tick_value=1.0,
    volume_min=0.01, volume_max=50.0, volume_step=0.01, trade_stops_level=10,
)
GOLD = SymbolSpec(
    name="GOLD", digits=2, point=0.01, trade_tick_size=0.01, trade_tick_value=1.0,
    volume_min=0.01, volume_max=20.0, volume_step=0.01, trade_stops_level=0,
)
ACCOUNT = AccountState(login=1, balance=10_000, equity=10_000, margin=0, margin_free=10_000,
                       margin_level=0, currency="USD", leverage=100)
TICK = Tick(symbol="EURUSD", bid=1.10000, ask=1.10010, time=0)  # spread 10 pts
ATR = 0.0020


def cfg(**kw) -> RiskConfig:
    base = dict(risk_per_trade_pct=1.0, max_spread_points=20, max_spread_atr_ratio=0.25, min_confidence=0.6,
                min_rr=1.5, max_sl_atr_mult=5, min_sl_atr_mult=0.3, max_open_positions=3,
                max_positions_per_symbol=1, daily_max_loss_pct=3.0, max_margin_usage_pct=50)
    base.update(kw)
    return RiskConfig(**base)


def buy(sl=1.09800, tp=1.10400, conf=0.8) -> AIDecision:
    return AIDecision(action=Action.BUY, confidence=conf, stop_loss=sl, take_profit=tp, reasoning="test")


async def evaluate(decision, *, config=None, spec=EURUSD, tick=TICK, account=ACCOUNT, positions=(),
                   dsb=10_000.0, atr=ATR, tick_age=0.0, loss_fn=None, margin_fn=None):
    rm = RiskManager(config or cfg())
    return await rm.evaluate(decision, spec, tick, account, list(positions), dsb, atr, tick_age,
                             loss_per_lot_fn=loss_fn, margin_fn=margin_fn)


# ---------------------------------------------------------------- normalize
@pytest.mark.parametrize("raw,expected", [(0.4999, 0.49), (0.01, 0.01), (0.009, 0.0), (123.0, 50.0), (1.005, 1.0)])
def test_normalize_volume(raw, expected):
    assert normalize_volume(EURUSD, raw) == expected


def test_normalize_volume_step_01():
    spec = SymbolSpec("CRYPTO_STEP01", 2, 0.01, 0.01, 0.01, volume_min=0.1, volume_max=100, volume_step=0.1)
    assert normalize_volume(spec, 0.37) == 0.3
    assert normalize_volume(spec, 0.09) == 0.0


def test_normalize_price():
    assert normalize_price(EURUSD, 1.098004) == 1.09800
    spec = SymbolSpec("IDX", 1, 0.1, 0.5, 1.0, 1, 100, 1)  # tick size 0.5
    assert normalize_price(spec, 100.3) == 100.5


# ---------------------------------------------------------------- happy path
async def test_approved_buy_volume_by_risk():
    p = await evaluate(buy())
    # SL distance = 1.10010 - 1.09800 = 0.0021 = 210 ticks * $1 = $210/lot ; risk $100 -> 0.47 lot
    assert p.approved, p.reasons
    assert p.side is Side.BUY
    assert p.volume == 0.47
    assert p.risk_amount == pytest.approx(98.7, abs=0.1)
    assert p.rr == pytest.approx(1.86, abs=0.01)


async def test_uses_loss_per_lot_fn_when_available():
    async def loss_fn(side, entry, sl):
        return 420.0  # cặp chéo – tick_value khác
    p = await evaluate(buy(), loss_fn=loss_fn)
    assert p.approved
    assert p.volume == 0.23


async def test_approved_sell():
    d = AIDecision(action=Action.SELL, confidence=0.9, stop_loss=1.10200, take_profit=1.09700)
    p = await evaluate(d)
    assert p.approved, p.reasons
    assert p.entry == TICK.bid


# ---------------------------------------------------------------- blocks
async def test_hold_is_not_tradable():
    p = await evaluate(AIDecision.hold("x"))
    assert not p.approved


async def test_low_confidence_blocked():
    p = await evaluate(buy(conf=0.3))
    assert not p.approved and any("confidence" in r for r in p.reasons)


async def test_spread_too_wide_blocked():
    wide = Tick("EURUSD", 1.10000, 1.10050, 0)  # 50 pts
    p = await evaluate(buy(), tick=wide)
    assert any("spread" in r for r in p.reasons)


async def test_spread_override_per_symbol():
    wide = Tick("EURUSD", 1.10000, 1.10030, 0)
    p = await evaluate(buy(), tick=wide, config=cfg(spread_overrides={"EURUSD": 40}, max_spread_atr_ratio=0))
    assert p.approved, p.reasons


async def test_spread_vs_atr_blocked():
    p = await evaluate(buy(sl=1.09995, tp=1.10030), atr=0.0002,
                       config=cfg(min_sl_atr_mult=0, min_rr=0))  # spread 0.0001 > 0.25*0.0002
    assert any("x ATR" in r for r in p.reasons)


async def test_missing_sl_blocked():
    p = await evaluate(buy(sl=None))
    assert not p.approved and any("stop loss" in r for r in p.reasons)


async def test_sl_fallback_atr():
    p = await evaluate(buy(sl=None, tp=1.10600), config=cfg(sl_fallback_atr_mult=1.5))
    assert p.approved, p.reasons
    assert p.sl == pytest.approx(1.10010 - 0.003, abs=1e-5)


async def test_sl_wrong_side_blocked():
    p = await evaluate(buy(sl=1.10100))
    assert not p.approved and any("phải <" in r for r in p.reasons)


async def test_sl_inside_stops_level_blocked():
    spec = SymbolSpec(**{**EURUSD.__dict__, "trade_stops_level": 500})  # 0.005
    p = await evaluate(buy(), spec=spec)
    assert any("stops_level" in r for r in p.reasons)


async def test_sl_too_far_in_atr_blocked():
    p = await evaluate(buy(sl=1.08000, tp=1.14000))
    assert any("quá xa" in r for r in p.reasons)


async def test_tp_wrong_side_blocked():
    p = await evaluate(buy(tp=1.09900))
    assert any("TP" in r for r in p.reasons)


async def test_rr_too_low_blocked():
    p = await evaluate(buy(tp=1.10200))
    assert any("R:R" in r for r in p.reasons)


async def test_daily_loss_limit_blocked():
    acc = AccountState(1, 10_000, 9_650, 0, 9_650, 0, "USD", 100)  # -3.5%
    p = await evaluate(buy(), account=acc)
    assert any("daily loss" in r for r in p.reasons)
    assert RiskManager(cfg()).daily_loss_hit(acc, 10_000)


async def test_max_positions_blocked():
    pos = [Position(i, f"SYM{i}", Side.BUY, 0.1, 1, 1, 0, 0, 0) for i in range(3)]
    p = await evaluate(buy(), positions=pos)
    assert any("lệnh mở" in r for r in p.reasons)


async def test_max_per_symbol_blocked():
    pos = [Position(1, "EURUSD", Side.SELL, 0.1, 1.1, 1.1, 0, 0, 0)]
    p = await evaluate(buy(), positions=pos)
    assert any("/symbol" in r for r in p.reasons)


async def test_trade_mode_blocks():
    p = await evaluate(buy(), spec=SymbolSpec(**{**EURUSD.__dict__, "trade_mode": TRADE_MODE_CLOSEONLY}))
    assert any("trade_mode" in r for r in p.reasons)
    d = AIDecision(action=Action.SELL, confidence=0.9, stop_loss=1.10200, take_profit=1.09700)
    p = await evaluate(d, spec=SymbolSpec(**{**EURUSD.__dict__, "trade_mode": TRADE_MODE_LONGONLY}))
    assert any("long-only" in r for r in p.reasons)


async def test_stale_tick_blocked():
    p = await evaluate(buy(), tick_age=999)
    assert any("tick cũ" in r for r in p.reasons)


async def test_volume_below_min_blocked():
    small = AccountState(1, 50, 50, 0, 50, 0, "USD", 100)
    p = await evaluate(buy(), account=small, dsb=50)
    assert not p.approved and any("volume_min" in r for r in p.reasons)


async def test_margin_check_blocked():
    async def margin_fn(side, vol, price):
        return 9_000.0
    p = await evaluate(buy(), margin_fn=margin_fn)
    assert any("margin" in r for r in p.reasons)


async def test_lot_cap():
    p = await evaluate(buy(), config=cfg(max_lot_cap=0.1))
    assert p.approved and p.volume == 0.1


async def test_gold_sizing():
    tick = Tick("GOLD", 2400.00, 2400.30, 0)  # 30 pts spread
    d = AIDecision(action=Action.BUY, confidence=0.8, stop_loss=2390.0, take_profit=2420.0)
    p = await evaluate(d, spec=GOLD, tick=tick, atr=5.0, config=cfg(max_spread_points=50))
    # SL dist 10.30 -> 1030 ticks * $1 = $1030/lot ; risk $100 -> 0.09 lot
    assert p.approved, p.reasons
    assert p.volume == 0.09


# ---------------------------------------------------------------- tài khoản nhỏ ($100) – spec thật của FxPro
BTC = SymbolSpec("BITCOIN", 2, 0.01, 0.01, 0.01, 0.01, 500, 0.01, trade_stops_level=20000)
GOLDGR = SymbolSpec("GOLDgr", 3, 0.001, 0.001, 0.001, 5.0, 500, 1.0)
SMALL = AccountState(1, 100, 100, 0, 100, 0, "USD", 100)


async def test_small_account_blocked_by_default():
    d = AIDecision(action=Action.BUY, confidence=0.8, stop_loss=61400.0, take_profit=62700.0)
    p = await evaluate(d, spec=BTC, tick=Tick("BITCOIN", 61890.0, 61890.0, 0), account=SMALL, dsb=100,
                       atr=355.0, config=cfg(max_spread_points=5000))
    assert not p.approved and any("volume_min" in r for r in p.reasons)


async def test_small_account_min_lot_bitcoin():
    # SL $490 -> 0.01 BTC lỗ $4.90 = 4.9% <= trần 5%
    d = AIDecision(action=Action.BUY, confidence=0.8, stop_loss=61400.0, take_profit=62700.0)
    p = await evaluate(d, spec=BTC, tick=Tick("BITCOIN", 61890.0, 61890.0, 0), account=SMALL, dsb=100,
                       atr=355.0, config=cfg(max_spread_points=5000, min_lot_max_risk_pct=5))
    assert p.approved, p.reasons
    assert p.volume == 0.01 and p.risk_amount == pytest.approx(4.9) and p.risk_pct == pytest.approx(4.9)
    assert p.notes and "lot tối thiểu" in p.notes[0]


async def test_small_account_over_cap_still_blocked():
    # SL $700 -> lỗ $7 = 7% > trần 5%
    d = AIDecision(action=Action.BUY, confidence=0.8, stop_loss=61190.0, take_profit=63000.0)
    p = await evaluate(d, spec=BTC, tick=Tick("BITCOIN", 61890.0, 61890.0, 0), account=SMALL, dsb=100,
                       atr=355.0, config=cfg(max_spread_points=5000, min_lot_max_risk_pct=5))
    assert not p.approved and any("trần 5" in r for r in p.reasons)


async def test_small_account_goldgr():
    # 5 gram, SL 0.6 USD/gram -> lỗ $3 = 3%
    d = AIDecision(action=Action.SELL, confidence=0.8, stop_loss=113.900, take_profit=112.100)
    p = await evaluate(d, spec=GOLDGR, tick=Tick("GOLDgr", 113.290, 113.302, 0), account=SMALL, dsb=100,
                       atr=0.62, config=cfg(max_spread_points=30, min_lot_max_risk_pct=5))
    assert p.approved, p.reasons
    assert p.volume == 5.0 and p.risk_amount == pytest.approx(3.05, abs=0.01)


def test_max_affordable_sl_distance():
    rm = RiskManager(cfg(risk_per_trade_pct=1, min_lot_max_risk_pct=5))
    assert rm.max_affordable_sl_distance(BTC, 100) == pytest.approx(500.0)   # $5 / $0.01 mỗi $1
    assert rm.max_affordable_sl_distance(GOLD, 100) == pytest.approx(5.0)    # 1 oz: $5 / $1
    assert rm.max_affordable_sl_distance(GOLDGR, 100) == pytest.approx(1.0)  # 5 g: $5 / $5
