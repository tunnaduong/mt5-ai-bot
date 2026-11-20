"""Test thống kê hiệu suất tích luỹ trong Storage."""
from __future__ import annotations

import pytest

from models import ClosedTrade
from storage import Storage


def add(db: Storage, ticket: int, symbol: str, profit: float, day: str) -> None:
    db.log_close(ClosedTrade(ticket, symbol, "BUY", 0.01, 1, 1, profit, "TP" if profit > 0 else "SL", 100))
    db._exec("UPDATE closed_trades SET day=? WHERE ticket=?", (day, ticket))


def test_performance_metrics():
    db = Storage(":memory:")
    for i, (sym, pnl, day) in enumerate([
        ("BITCOIN", 10, "2026-11-01"), ("BITCOIN", -5, "2026-11-01"), ("GOLD", -5, "2026-11-02"),
        ("GOLD", -4, "2026-11-02"), ("BITCOIN", 12, "2026-11-03"),
    ]):
        add(db, i, sym, pnl, day)
    st = db.performance()
    assert st["trades"] == 5 and st["wins"] == 2 and st["winrate"] == pytest.approx(40.0)
    assert st["pnl"] == pytest.approx(8.0)
    assert st["profit_factor"] == pytest.approx(22 / 14)
    assert st["max_drawdown"] == pytest.approx(14.0)  # đỉnh +10 -> đáy -4
    assert st["max_losing_streak"] == 3
    assert st["by_symbol"]["BITCOIN"]["pnl"] == pytest.approx(17.0)
    assert list(st["by_day"]) == ["2026-11-01", "2026-11-02", "2026-11-03"]
    assert db.performance("2026-11-03")["trades"] == 1


def test_performance_empty():
    st = Storage(":memory:").performance()
    assert st["trades"] == 0 and st["first_day"] is None
