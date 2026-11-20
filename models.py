"""Các kiểu dữ liệu trung lập (không phụ thuộc MetaTrader5) dùng chung giữa các module.

Việc tách riêng giúp `risk_manager` và test chạy được trên mọi OS mà không cần package MT5.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

# Giá trị SYMBOL_TRADE_MODE_* của MT5
TRADE_MODE_DISABLED = 0
TRADE_MODE_LONGONLY = 1
TRADE_MODE_SHORTONLY = 2
TRADE_MODE_CLOSEONLY = 3
TRADE_MODE_FULL = 4

# Giá trị ACCOUNT_TRADE_MODE_*
ACCOUNT_DEMO = 0
ACCOUNT_CONTEST = 1
ACCOUNT_REAL = 2


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"

    @property
    def opposite(self) -> "Side":
        return Side.SELL if self is Side.BUY else Side.BUY


@dataclass(frozen=True)
class SymbolSpec:
    """Thuộc tính giao dịch của một symbol, lấy động từ `mt5.symbol_info`."""

    name: str
    digits: int
    point: float
    trade_tick_size: float
    trade_tick_value: float
    volume_min: float
    volume_max: float
    volume_step: float
    trade_stops_level: int = 0
    trade_freeze_level: int = 0
    trade_mode: int = TRADE_MODE_FULL
    filling_mode: int = 1
    currency_profit: str = ""
    description: str = ""

    @property
    def min_stop_distance(self) -> float:
        """Khoảng cách tối thiểu (theo giá) giữa giá hiện tại và SL/TP mà broker chấp nhận."""
        return max(self.trade_stops_level, self.trade_freeze_level) * self.point


@dataclass(frozen=True)
class Tick:
    symbol: str
    bid: float
    ask: float
    time: int  # epoch theo giờ server broker

    def spread_points(self, point: float) -> float:
        return round((self.ask - self.bid) / point, 1) if point > 0 else 0.0


@dataclass(frozen=True)
class AccountState:
    login: int
    balance: float
    equity: float
    margin: float
    margin_free: float
    margin_level: float
    currency: str
    leverage: int
    server: str = ""
    trade_mode: int = ACCOUNT_DEMO
    trade_allowed: bool = True

    @property
    def is_real(self) -> bool:
        return self.trade_mode == ACCOUNT_REAL


@dataclass(frozen=True)
class Position:
    ticket: int
    symbol: str
    side: Side
    volume: float
    price_open: float
    price_current: float
    sl: float
    tp: float
    profit: float
    swap: float = 0.0
    magic: int = 0
    comment: str = ""
    time: int = 0


@dataclass
class OrderResult:
    success: bool
    retcode: int
    comment: str
    dry_run: bool = False
    ticket: int | None = None
    price: float | None = None
    volume: float | None = None
    request: dict = field(default_factory=dict)


@dataclass
class ClosedTrade:
    ticket: int
    symbol: str
    side: str
    volume: float
    open_price: float
    close_price: float
    profit: float  # net: profit + swap + commission + fee
    reason: str
    balance_after: float

    @property
    def balance_change_pct(self) -> float:
        before = self.balance_after - self.profit
        return (self.profit / before * 100.0) if before > 0 else 0.0
