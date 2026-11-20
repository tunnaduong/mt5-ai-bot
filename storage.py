"""Lưu vết toàn bộ hoạt động vào SQLite: AI call, risk check, lệnh, lệnh đóng, sự kiện."""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from models import ClosedTrade

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ai_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL, symbol TEXT, ok INTEGER, attempts INTEGER, latency_ms INTEGER,
    request_json TEXT, response_text TEXT, decision_json TEXT, error TEXT
);
CREATE TABLE IF NOT EXISTS risk_checks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL, symbol TEXT, action TEXT, confidence REAL, approved INTEGER,
    volume REAL, entry REAL, sl REAL, tp REAL, risk_amount REAL, rr REAL, reasons TEXT
);
CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL, symbol TEXT, side TEXT, volume REAL, price REAL, sl REAL, tp REAL,
    dry_run INTEGER, success INTEGER, retcode INTEGER, comment TEXT, ticket INTEGER, request_json TEXT
);
CREATE TABLE IF NOT EXISTS closed_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL, day TEXT NOT NULL, ticket INTEGER UNIQUE, symbol TEXT, side TEXT, volume REAL,
    open_price REAL, close_price REAL, profit REAL, reason TEXT, balance_after REAL
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL, level TEXT, kind TEXT, message TEXT
);
CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT);
"""


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _ts() -> str:
    return utc_now().isoformat(timespec="seconds")


def _dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


class Storage:
    """Wrapper SQLite đơn giản, thread-safe (mỗi thao tác rất nhanh nên gọi đồng bộ)."""

    def __init__(self, path: str | Path) -> None:
        path = Path(path)
        if str(path) != ":memory:":
            path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def _exec(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self._conn.execute(sql, params)
            self._conn.commit()
            return cur

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------------ writes
    def log_ai_call(
        self,
        symbol: str,
        ok: bool,
        attempts: int,
        latency_ms: int,
        request: Any,
        response_text: str | None,
        decision: Any,
        error: str | None,
    ) -> None:
        self._exec(
            "INSERT INTO ai_calls(ts,symbol,ok,attempts,latency_ms,request_json,response_text,decision_json,error)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (_ts(), symbol, int(ok), attempts, latency_ms, _dumps(request), response_text, _dumps(decision), error),
        )

    def log_risk(
        self,
        symbol: str,
        action: str,
        confidence: float,
        approved: bool,
        volume: float,
        entry: float | None,
        sl: float | None,
        tp: float | None,
        risk_amount: float,
        rr: float | None,
        reasons: list[str],
    ) -> None:
        self._exec(
            "INSERT INTO risk_checks(ts,symbol,action,confidence,approved,volume,entry,sl,tp,risk_amount,rr,reasons)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (_ts(), symbol, action, confidence, int(approved), volume, entry, sl, tp, risk_amount, rr, _dumps(reasons)),
        )

    def log_order(
        self,
        symbol: str,
        side: str,
        volume: float,
        price: float | None,
        sl: float | None,
        tp: float | None,
        dry_run: bool,
        success: bool,
        retcode: int,
        comment: str,
        ticket: int | None,
        request: dict,
    ) -> None:
        self._exec(
            "INSERT INTO orders(ts,symbol,side,volume,price,sl,tp,dry_run,success,retcode,comment,ticket,request_json)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (_ts(), symbol, side, volume, price, sl, tp, int(dry_run), int(success), retcode, comment, ticket,
             _dumps(request)),
        )

    def log_close(self, trade: ClosedTrade) -> None:
        now = utc_now()
        self._exec(
            "INSERT OR IGNORE INTO closed_trades(ts,day,ticket,symbol,side,volume,open_price,close_price,profit,"
            "reason,balance_after) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (now.isoformat(timespec="seconds"), now.date().isoformat(), trade.ticket, trade.symbol, trade.side,
             trade.volume, trade.open_price, trade.close_price, trade.profit, trade.reason, trade.balance_after),
        )

    def log_event(self, level: str, kind: str, message: str) -> None:
        self._exec("INSERT INTO events(ts,level,kind,message) VALUES (?,?,?,?)", (_ts(), level, kind, message))

    # ------------------------------------------------------------------ kv
    def get_kv(self, key: str) -> str | None:
        row = self._exec("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        return row["value"] if row else None

    def set_kv(self, key: str, value: str) -> None:
        self._exec("INSERT INTO kv(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                   (key, value))

    # ------------------------------------------------------------------ reads
    def daily_stats(self, day: str) -> dict[str, float]:
        """Thống kê lệnh đã đóng trong ngày (UTC) `day` dạng YYYY-MM-DD."""
        row = self._exec(
            "SELECT COUNT(*) AS n, SUM(CASE WHEN profit > 0 THEN 1 ELSE 0 END) AS wins,"
            " COALESCE(SUM(profit),0) AS pnl, COALESCE(MAX(profit),0) AS best, COALESCE(MIN(profit),0) AS worst"
            " FROM closed_trades WHERE day=?",
            (day,),
        ).fetchone()
        n = int(row["n"] or 0)
        wins = int(row["wins"] or 0)
        return {
            "trades": n,
            "wins": wins,
            "losses": n - wins,
            "winrate": (wins / n * 100.0) if n else 0.0,
            "pnl": float(row["pnl"] or 0.0),
            "best": float(row["best"] or 0.0),
            "worst": float(row["worst"] or 0.0),
        }

    def performance(self, day_from: str | None = None) -> dict[str, Any]:
        """Thống kê tích luỹ các lệnh đã đóng từ ngày `day_from` (YYYY-MM-DD, UTC); None = toàn bộ.

        Trả về tổng quan, theo symbol và theo ngày, kèm profit factor, chuỗi thua dài nhất và
        max drawdown của đường P&L realized.
        """
        sql = "SELECT day, symbol, profit FROM closed_trades"
        params: tuple = ()
        if day_from:
            sql += " WHERE day >= ?"
            params = (day_from,)
        rows = self._exec(sql + " ORDER BY ts, id", params).fetchall()
        profits = [float(r["profit"]) for r in rows]

        def summary(values: list[float]) -> dict[str, float]:
            wins = [v for v in values if v > 0]
            losses = [v for v in values if v <= 0]
            gross_win, gross_loss = sum(wins), -sum(losses)
            return {
                "trades": len(values),
                "wins": len(wins),
                "losses": len(losses),
                "winrate": len(wins) / len(values) * 100.0 if values else 0.0,
                "pnl": sum(values),
                "avg_win": gross_win / len(wins) if wins else 0.0,
                "avg_loss": -gross_loss / len(losses) if losses else 0.0,
                "profit_factor": gross_win / gross_loss if gross_loss > 0 else (float("inf") if wins else 0.0),
            }

        result: dict[str, Any] = summary(profits)
        equity = peak = max_dd = 0.0
        streak = max_streak = 0
        for v in profits:
            equity += v
            peak = max(peak, equity)
            max_dd = max(max_dd, peak - equity)
            streak = streak + 1 if v <= 0 else 0
            max_streak = max(max_streak, streak)
        result["max_drawdown"] = max_dd
        result["max_losing_streak"] = max_streak

        by_symbol: dict[str, list[float]] = {}
        by_day: dict[str, list[float]] = {}
        for r in rows:
            by_symbol.setdefault(r["symbol"], []).append(float(r["profit"]))
            by_day.setdefault(r["day"], []).append(float(r["profit"]))
        result["by_symbol"] = {k: summary(v) for k, v in sorted(by_symbol.items())}
        result["by_day"] = {k: summary(v) for k, v in sorted(by_day.items())}
        result["first_day"] = rows[0]["day"] if rows else None
        return result

    def orders_today_count(self, day: str) -> int:
        row = self._exec("SELECT COUNT(*) AS n FROM orders WHERE substr(ts,1,10)=? AND success=1", (day,)).fetchone()
        return int(row["n"] or 0)
