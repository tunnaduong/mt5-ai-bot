"""Test luồng end-to-end của TradingBot với MT5 giả lập + AI giả lập (không cần Windows/MT5)."""
from __future__ import annotations

import json

import httpx
import numpy as np
import pandas as pd
import pytest

from ai_client import AIClient
from bot import TradingBot, build_risk_config
from config import Settings
from models import AccountState, ClosedTrade, OrderResult, Position, Side, SymbolSpec, Tick
from risk_manager import RiskManager
from storage import Storage


class FakeMT5:
    """Giả lập các coroutine public của MT5Client."""

    def __init__(self) -> None:
        self.dry_run = True
        self.magic = 777
        self.connected = True
        self.opened: list[tuple] = []
        self.closed: list[int] = []
        self.pos: list[Position] = []
        self.acc = AccountState(1, 10_000, 10_000, 0, 10_000, 0, "USD", 100)
        self.spec = SymbolSpec("EURUSD", 5, 0.00001, 0.00001, 1.0, 0.01, 50, 0.01, trade_stops_level=10)
        n = 260
        c = 1.1 + np.linspace(0, 0.01, n)
        self.df = pd.DataFrame({"time": np.arange(n) * 900, "open": c - 0.0002, "high": c + 0.0010,
                                "low": c - 0.0010, "close": c, "tick_volume": np.full(n, 500)})

    async def ensure_connected(self): return True
    async def account(self): return self.acc
    async def symbol_spec(self, s): return self.spec
    async def tick(self, s): return Tick(s, 1.11000, 1.11010, 123)
    async def bars(self, s, tf, n): return self.df
    async def positions(self, symbol=None, magic_only=False): return list(self.pos)
    async def loss_per_lot(self, s, side, e, sl): return None
    async def margin_required(self, s, side, v, p): return 100.0

    async def open_position(self, spec, side, vol, sl, tp, comment=""):
        self.opened.append((side, vol, sl, tp))
        return OrderResult(True, 0, "DRY-RUN ok", True, price=1.1101, volume=vol)

    async def close_position(self, p, comment="", force=False):
        self.closed.append(p.ticket)
        return OrderResult(True, 10009, "done", False)

    async def closed_trade_info(self, p):
        return ClosedTrade(p.ticket, p.symbol, p.side.value, p.volume, p.price_open, 1.1150, 55.0, "Take Profit", 10_055)


class FakeTG:
    def __init__(self): self.msgs: list[str] = []
    def notify(self, text, keyboard=None): self.msgs.append(text)


def ai_returning(decision: dict) -> AIClient:
    body = {"choices": [{"message": {"content": json.dumps(decision)}}]}
    return AIClient("https://ai.test", transport=httpx.MockTransport(lambda r: httpx.Response(200, json=body)),
                    max_retries=1)


def make_bot(decision: dict, **settings_kw):
    s = Settings(_env_file=None, symbols="EURUSD", context_timeframes="H1", risk_per_trade_pct=1.0, **settings_kw)
    mt5, tg = FakeMT5(), FakeTG()
    bot = TradingBot(s, mt5, ai_returning(decision), RiskManager(build_risk_config(s)), Storage(":memory:"), tg)
    return bot, mt5, tg


BUY = {"action": "BUY", "confidence": 0.8, "stop_loss": 1.1080, "take_profit": 1.1150, "reasoning": "trend up"}


async def test_buy_flow_opens_and_notifies():
    bot, mt5, tg = make_bot(BUY)
    await bot.monitor_once()
    await bot.trading_once(force=True)
    assert len(mt5.opened) == 1
    side, vol, sl, tp = mt5.opened[0]
    assert side is Side.BUY and sl == 1.108 and vol > 0
    assert any("DRY-RUN" in m for m in tg.msgs)
    # cùng nến -> không phân tích lại
    await bot.trading_once()
    assert len(mt5.opened) == 1


async def test_paused_does_not_trade():
    bot, mt5, _ = make_bot(BUY)
    bot.pause()
    await bot.trading_once(force=True)
    assert mt5.opened == []


async def test_risk_block_notifies():
    bot, mt5, tg = make_bot({**BUY, "stop_loss": None})
    await bot.trading_once(force=True)
    assert mt5.opened == [] and any("Risk BLOCK" in m for m in tg.msgs)


async def test_close_detection_and_daily_summary():
    bot, mt5, tg = make_bot(BUY, daily_summary_hour_utc=0)
    mt5.pos = [Position(5, "EURUSD", Side.BUY, 0.1, 1.11, 1.115, 1.10, 1.115, 50.0, magic=777)]
    await bot.monitor_once()
    mt5.pos = []
    await bot.monitor_once()
    assert any("Take Profit" in m for m in tg.msgs)
    assert bot.db.daily_stats(bot.today())["trades"] == 1
    assert any("TỔNG KẾT" in m for m in tg.msgs)


async def test_ai_close_closes_bot_positions_only():
    bot, mt5, _ = make_bot({"action": "CLOSE", "confidence": 0.9, "reasoning": "invalidated"})
    mt5.pos = [Position(1, "EURUSD", Side.BUY, 0.1, 1.11, 1.11, 1.1, 1.12, -5, magic=777),
               Position(2, "EURUSD", Side.BUY, 0.1, 1.11, 1.11, 1.1, 1.12, -5, magic=0)]
    await bot.trading_once(force=True)
    assert mt5.closed == [1]


async def test_daily_loss_alert_once():
    bot, mt5, tg = make_bot(BUY)
    await bot.monitor_once()  # ghi day_start_balance = 10000
    mt5.acc = AccountState(1, 10_000, 9_600, 0, 9_600, 0, "USD", 100)
    await bot.monitor_once()
    await bot.monitor_once()
    assert sum("DAILY LOSS" in m for m in tg.msgs) == 1


async def test_controller_texts():
    bot, mt5, _ = make_bot(BUY)
    mt5.pos = [Position(9, "GOLD", Side.SELL, 0.05, 2400, 2390, 2410, 2380, 50.0, magic=777)]
    assert "Balance" in await bot.status_text()
    assert "GOLD" in await bot.positions_text()
    assert "✅ 1" in await bot.close_all_positions()
    text = await bot.analyze_text("EURUSD")
    assert "BUY" in text and "Risk" in text and mt5.opened == []


@pytest.fixture(autouse=True)
def _no_file_logs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
