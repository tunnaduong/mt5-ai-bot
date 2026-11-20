"""Cấu hình tập trung cho bot, đọc từ biến môi trường / file `.env` qua pydantic-settings.

Mọi tham số nhạy cảm (mật khẩu, API key, token) dùng `SecretStr` để không bị lộ khi log.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


class Settings(BaseSettings):
    """Toàn bộ cấu hình runtime. Tên field (không phân biệt hoa/thường) = tên biến môi trường."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False
    )

    # ---------------- MetaTrader 5 ----------------
    mt5_login: int = 0
    mt5_password: SecretStr = SecretStr("")
    mt5_server: str = ""  # ví dụ: FxPro-MT5 Demo
    mt5_path: str | None = None  # đường dẫn terminal64.exe (tùy chọn)
    mt5_magic: int = 20261120
    mt5_deviation: int = Field(20, ge=0, description="Độ trượt giá tối đa (points)")
    mt5_order_retries: int = Field(3, ge=1, le=10)

    # ---------------- AI endpoint ----------------
    ai_endpoint_url: str = "https://api.openai.com/v1/chat/completions"
    ai_api_key: SecretStr = SecretStr("")
    ai_model: str = "gpt-4o-mini"
    ai_api_style: Literal["openai", "custom"] = "openai"
    ai_timeout_sec: float = Field(30.0, gt=0)
    ai_max_retries: int = Field(3, ge=1, le=10)
    ai_temperature: float = Field(0.2, ge=0, le=2)
    ai_json_mode: bool = True  # gửi response_format={"type":"json_object"} (OpenAI-compatible)
    ai_failure_alert_threshold: int = Field(3, ge=1)
    ai_reasoning_language: str = "Vietnamese"

    # ---------------- Telegram ----------------
    telegram_bot_token: SecretStr = SecretStr("")
    telegram_admin_chat_id: int = 0
    telegram_min_interval_sec: float = Field(1.1, ge=0.05)

    # ---------------- Trading loop ----------------
    symbols: str = "EURUSD,GOLD"
    timeframe: str = "M15"
    context_timeframes: str = "H1,H4"
    bars_count: int = Field(250, ge=60, le=5000)
    loop_interval_sec: float = Field(10.0, gt=0)
    monitor_interval_sec: float = Field(5.0, gt=0)
    dry_run: bool = True
    allow_real_account: bool = False  # phải bật thêm cờ này mới cho phép lệnh thật trên tài khoản REAL
    notify_min_confidence: float = Field(0.6, ge=0, le=1)
    daily_summary_hour_utc: int = Field(21, ge=0, le=23)

    # ---------------- Risk ----------------
    risk_per_trade_pct: float = Field(0.5, gt=0, le=5)
    max_open_positions: int = Field(3, ge=1)
    max_positions_per_symbol: int = Field(1, ge=1)
    daily_max_loss_pct: float = Field(3.0, gt=0, le=50)
    daily_loss_close_all: bool = False
    max_spread_points: float = Field(30.0, gt=0)
    max_spread_overrides: str = ""  # "GOLD:50,BITCOIN:5000"
    max_spread_atr_ratio: float = Field(0.25, ge=0)  # 0 = tắt
    min_confidence: float = Field(0.65, ge=0, le=1)
    min_rr: float = Field(1.2, ge=0)  # 0 = không bắt buộc TP
    max_sl_atr_mult: float = Field(5.0, ge=0)  # 0 = tắt
    min_sl_atr_mult: float = Field(0.3, ge=0)
    sl_fallback_atr_mult: float = Field(0.0, ge=0)  # 0 = AI bắt buộc phải gửi SL
    max_tick_age_sec: int = Field(180, ge=5)
    max_margin_usage_pct: float = Field(30.0, gt=0, le=100)
    max_lot_cap: float = Field(0.0, ge=0)  # 0 = không giới hạn thêm
    min_lot_max_risk_pct: float = Field(0.0, ge=0, le=10)  # tài khoản nhỏ; 0 = tắt

    # ---------------- Storage / logging ----------------
    db_path: str = "data/bot.db"
    log_dir: str = "logs"
    log_level: str = "INFO"

    @field_validator("timeframe")
    @classmethod
    def _upper_tf(cls, v: str) -> str:
        return v.strip().upper()

    @model_validator(mode="after")
    def _check_consistency(self) -> "Settings":
        if not self.symbol_list:
            raise ValueError("SYMBOLS không được rỗng")
        return self

    # ---------------- Derived helpers ----------------
    @property
    def symbol_list(self) -> list[str]:
        return _split_csv(self.symbols)

    @property
    def context_timeframe_list(self) -> list[str]:
        return [tf.upper() for tf in _split_csv(self.context_timeframes)]

    @property
    def spread_overrides(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for item in _split_csv(self.max_spread_overrides):
            name, _, value = item.partition(":")
            if name and value:
                result[name.strip()] = float(value)
        return result

    @property
    def telegram_enabled(self) -> bool:
        return bool(self.telegram_bot_token.get_secret_value()) and self.telegram_admin_chat_id != 0


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Trả về Settings singleton (đọc `.env` một lần)."""
    return Settings()
