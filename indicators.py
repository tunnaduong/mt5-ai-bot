"""Chỉ báo kỹ thuật (EMA, RSI, ATR, cấu trúc nến) và hàm tóm tắt gọn cho AI.

Đầu vào là DataFrame OHLCV với các cột `time, open, high, low, close, tick_volume`
(đúng định dạng `mt5.copy_rates_from_pos`) và CHỈ chứa nến đã đóng.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average (chuẩn, adjust=False giống MT5)."""
    return series.ewm(span=period, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI theo phương pháp làm mượt Wilder."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    # avg_loss == 0 -> RSI = 100 (chỉ tăng)
    return out.where(avg_loss != 0.0, 100.0)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range theo Wilder."""
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def candle_pattern(df: pd.DataFrame) -> str:
    """Nhận diện mẫu nến đơn giản ở nến đóng gần nhất."""
    if len(df) < 2:
        return "n/a"
    c, p = df.iloc[-1], df.iloc[-2]
    body = abs(c.close - c.open)
    rng = max(c.high - c.low, 1e-12)
    upper = c.high - max(c.close, c.open)
    lower = min(c.close, c.open) - c.low
    if body / rng < 0.1:
        return "doji"
    if c.close > c.open and p.close < p.open and c.close >= p.open and c.open <= p.close:
        return "bullish_engulfing"
    if c.close < c.open and p.close > p.open and c.open >= p.close and c.close <= p.open:
        return "bearish_engulfing"
    if lower > 2 * body and upper < body:
        return "hammer/bullish_pin"
    if upper > 2 * body and lower < body:
        return "shooting_star/bearish_pin"
    return "bullish" if c.close > c.open else "bearish"


def _trend_label(close: float, e20: float, e50: float, e200: float | None) -> str:
    if e200 is not None and close > e20 > e50 > e200:
        return "strong_up"
    if e200 is not None and close < e20 < e50 < e200:
        return "strong_down"
    if close > e50 and e20 > e50:
        return "up"
    if close < e50 and e20 < e50:
        return "down"
    return "range"


def _r(x: float | None, digits: int) -> float | None:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return round(float(x), digits)


def _bar_time(df: pd.DataFrame) -> int | None:
    """Epoch (giờ server) của nến đóng gần nhất; hỗ trợ cột time dạng số hoặc datetime."""
    if "time" not in df:
        return None
    col = df["time"]
    if pd.api.types.is_numeric_dtype(col):
        return int(col.iloc[-1])
    return int(pd.Timestamp(col.iloc[-1]).timestamp())


def summarize(df: pd.DataFrame, timeframe: str, digits: int, n_recent: int = 5) -> dict[str, Any]:
    """Tóm tắt một khung thời gian thành dict nhỏ gọn (ít token) cho AI."""
    if df is None or len(df) < 30:
        return {"timeframe": timeframe, "error": "not_enough_bars"}

    close = df["close"]
    e20, e50 = ema(close, 20), ema(close, 50)
    e200 = ema(close, 200) if len(df) >= 200 else None
    r = rsi(close, 14)
    a = atr(df, 14)
    last = df.iloc[-1]

    lookback = df.tail(50)
    vol_avg = float(df["tick_volume"].tail(20).mean()) if "tick_volume" in df else 0.0
    rel_vol = float(last["tick_volume"]) / vol_avg if vol_avg > 0 and "tick_volume" in df else None

    recent = [
        [_r(row.open, digits), _r(row.high, digits), _r(row.low, digits), _r(row.close, digits)]
        for row in df.tail(n_recent).itertuples()
    ]
    e200_last = float(e200.iloc[-1]) if e200 is not None else None

    return {
        "timeframe": timeframe,
        "last_bar_time": _bar_time(df),
        "close": _r(last.close, digits),
        "ema20": _r(e20.iloc[-1], digits),
        "ema50": _r(e50.iloc[-1], digits),
        "ema200": _r(e200_last, digits),
        "rsi14": _r(r.iloc[-1], 1),
        "rsi14_prev": _r(r.iloc[-2], 1),
        "atr14": _r(a.iloc[-1], digits),
        "trend": _trend_label(float(last.close), float(e20.iloc[-1]), float(e50.iloc[-1]), e200_last),
        "swing_high_50": _r(lookback["high"].max(), digits),
        "swing_low_50": _r(lookback["low"].min(), digits),
        "last_candle": candle_pattern(df),
        "rel_volume": _r(rel_vol, 2),
        "recent_ohlc": recent,
    }


def latest_atr(df: pd.DataFrame, period: int = 14) -> float | None:
    """ATR của nến đóng gần nhất (None nếu thiếu dữ liệu)."""
    if df is None or len(df) <= period:
        return None
    val = atr(df, period).iloc[-1]
    return None if np.isnan(val) else float(val)
