"""Wrapper bất đồng bộ cho package `MetaTrader5` (chỉ chạy trên Windows có terminal MT5).

Package MetaTrader5 là blocking và KHÔNG thread-safe, nên mọi lời gọi được dồn vào một
ThreadPoolExecutor 1 luồng duy nhất -> tuần tự, không chặn event loop asyncio.
"""
from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, TypeVar

import pandas as pd

from models import AccountState, ClosedTrade, OrderResult, Position, Side, SymbolSpec, Tick

try:  # pragma: no cover - phụ thuộc OS
    import MetaTrader5 as mt5  # type: ignore[import-not-found]
except ImportError:  # Linux/macOS: chỉ dùng được cho test với mock
    mt5 = None

log = logging.getLogger(__name__)
T = TypeVar("T")

TIMEFRAMES = ("M1", "M2", "M3", "M5", "M10", "M15", "M30", "H1", "H2", "H4", "H6", "H8", "H12", "D1", "W1", "MN1")

# Retcode MT5
RET_DONE, RET_PLACED, RET_DONE_PARTIAL = 10009, 10008, 10010
RETRYABLE_RETCODES = {10004, 10012, 10020, 10021, 10024, 10031}  # requote, timeout, price changed/off, too many, no conn
SUCCESS_RETCODES = {RET_DONE, RET_PLACED, RET_DONE_PARTIAL}

# Bit filling_mode của symbol (SYMBOL_FILLING_FOK=1, SYMBOL_FILLING_IOC=2)
SYMBOL_FILLING_FOK, SYMBOL_FILLING_IOC = 1, 2

DEAL_ENTRY_IN, DEAL_ENTRY_OUT, DEAL_ENTRY_INOUT, DEAL_ENTRY_OUT_BY = 0, 1, 2, 3
DEAL_REASONS = {0: "Client (manual)", 1: "Mobile", 2: "Web", 3: "Expert/Bot", 4: "Stop Loss", 5: "Take Profit",
                6: "Stop Out", 7: "Rollover", 8: "Variation margin", 9: "Split"}


class MT5Error(RuntimeError):
    pass


class MT5Client:
    """Kết nối MT5 + thao tác dữ liệu/lệnh. Tất cả public method đều là coroutine."""

    def __init__(
        self,
        login: int,
        password: str,
        server: str,
        path: str | None = None,
        magic: int = 20261120,
        deviation: int = 20,
        dry_run: bool = True,
        order_retries: int = 3,
        connect_retries: int = 3,
    ) -> None:
        if mt5 is None:
            raise MT5Error("Không import được package MetaTrader5 (chỉ hỗ trợ Windows: pip install MetaTrader5)")
        self.login, self._password, self.server, self.path = login, password, server, path
        self.magic, self.deviation, self.dry_run = magic, deviation, dry_run
        self.order_retries, self.connect_retries = order_retries, connect_retries
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mt5")
        self.connected = False

    # ================================================================== infra
    async def _run(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, lambda: fn(*args, **kwargs))

    @staticmethod
    def _last_error() -> str:
        code, msg = mt5.last_error()
        return f"[{code}] {msg}"

    @staticmethod
    def timeframe(name: str) -> int:
        name = name.upper()
        if name not in TIMEFRAMES:
            raise MT5Error(f"Timeframe không hợp lệ: {name}")
        return getattr(mt5, f"TIMEFRAME_{name}")

    # ================================================================== connection
    def _connect_sync(self) -> list[str]:
        kwargs: dict[str, Any] = {"timeout": 60_000}
        if self.login:
            kwargs.update(login=self.login, password=self._password, server=self.server)
        ok = mt5.initialize(self.path, **kwargs) if self.path else mt5.initialize(**kwargs)
        if not ok:
            raise MT5Error(f"mt5.initialize() thất bại: {self._last_error()}")
        warnings: list[str] = []
        term = mt5.terminal_info()
        acc = mt5.account_info()
        if term is None or acc is None:
            raise MT5Error(f"Không lấy được terminal/account info: {self._last_error()}")
        if not term.connected:
            warnings.append("Terminal chưa kết nối tới trade server")
        if not term.trade_allowed:
            warnings.append("Nút 'Algo Trading' trên terminal đang TẮT – lệnh sẽ bị từ chối")
        if not acc.trade_allowed:
            warnings.append("Tài khoản không được phép giao dịch")
        if not acc.trade_expert:
            warnings.append("Tài khoản không cho phép giao dịch tự động (trade_expert=False)")
        return warnings

    async def connect(self) -> list[str]:
        """Kết nối có retry. Trả về danh sách cảnh báo (algo trading tắt...)."""
        last: Exception | None = None
        for attempt in range(1, self.connect_retries + 1):
            try:
                warnings = await self._run(self._connect_sync)
                self.connected = True
                acc = await self.account()
                log.info("MT5 connected: login=%s server=%s balance=%.2f %s", acc.login, acc.server,
                         acc.balance, acc.currency)
                for w in warnings:
                    log.warning("MT5: %s", w)
                return warnings
            except MT5Error as exc:
                last = exc
                log.error("Kết nối MT5 lần %d/%d lỗi: %s", attempt, self.connect_retries, exc)
                await self._run(mt5.shutdown)
                await asyncio.sleep(min(2 ** attempt, 30))
        self.connected = False
        raise MT5Error(f"Không kết nối được MT5 sau {self.connect_retries} lần: {last}")

    async def ensure_connected(self) -> bool:
        """Kiểm tra kết nối; tự kết nối lại nếu mất. Trả về True nếu đang kết nối."""
        def _check() -> bool:
            term = mt5.terminal_info()
            return bool(term and term.connected and mt5.account_info() is not None)

        try:
            if await self._run(_check):
                self.connected = True
                return True
        except Exception as exc:  # noqa: BLE001
            log.warning("MT5 health check lỗi: %s", exc)
        log.warning("Mất kết nối MT5 – đang thử kết nối lại...")
        self.connected = False
        try:
            await self._run(mt5.shutdown)
            await self.connect()
            return True
        except MT5Error as exc:
            log.error("Reconnect MT5 thất bại: %s", exc)
            return False

    async def shutdown(self) -> None:
        try:
            await self._run(mt5.shutdown)
        finally:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self.connected = False

    # ================================================================== data
    async def account(self) -> AccountState:
        a = await self._run(mt5.account_info)
        if a is None:
            raise MT5Error(f"account_info() None: {self._last_error()}")
        return AccountState(
            login=a.login, balance=a.balance, equity=a.equity, margin=a.margin, margin_free=a.margin_free,
            margin_level=a.margin_level or 0.0, currency=a.currency, leverage=a.leverage, server=a.server,
            trade_mode=a.trade_mode, trade_allowed=bool(a.trade_allowed and a.trade_expert),
        )

    async def symbol_spec(self, symbol: str) -> SymbolSpec:
        def _get() -> Any:
            info = mt5.symbol_info(symbol)
            if info is None:
                return None
            if not info.visible and not mt5.symbol_select(symbol, True):
                return None
            return mt5.symbol_info(symbol)

        i = await self._run(_get)
        if i is None:
            raise MT5Error(f"Symbol '{symbol}' không tồn tại trên broker: {self._last_error()}")
        return SymbolSpec(
            name=i.name, digits=i.digits, point=i.point, trade_tick_size=i.trade_tick_size,
            trade_tick_value=i.trade_tick_value, volume_min=i.volume_min, volume_max=i.volume_max,
            volume_step=i.volume_step, trade_stops_level=i.trade_stops_level,
            trade_freeze_level=i.trade_freeze_level, trade_mode=i.trade_mode, filling_mode=i.filling_mode,
            currency_profit=i.currency_profit, description=i.description,
        )

    async def tick(self, symbol: str) -> Tick:
        t = await self._run(mt5.symbol_info_tick, symbol)
        if t is None:
            raise MT5Error(f"Không lấy được tick {symbol}: {self._last_error()}")
        return Tick(symbol=symbol, bid=t.bid, ask=t.ask, time=int(t.time))

    async def bars(self, symbol: str, timeframe: str, count: int) -> pd.DataFrame:
        """Lấy `count` nến ĐÃ ĐÓNG (bỏ nến hiện tại đang chạy, start_pos=1)."""
        rates = await self._run(mt5.copy_rates_from_pos, symbol, self.timeframe(timeframe), 1, count)
        if rates is None or len(rates) == 0:
            raise MT5Error(f"Không lấy được nến {symbol} {timeframe}: {self._last_error()}")
        return pd.DataFrame(rates)

    async def positions(self, symbol: str | None = None, magic_only: bool = False) -> list[Position]:
        raw = await self._run(lambda: mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get())
        if raw is None:
            raw = ()
        out = [
            Position(
                ticket=p.ticket, symbol=p.symbol, side=Side.BUY if p.type == 0 else Side.SELL, volume=p.volume,
                price_open=p.price_open, price_current=p.price_current, sl=p.sl, tp=p.tp, profit=p.profit,
                swap=p.swap, magic=p.magic, comment=p.comment, time=p.time,
            )
            for p in raw
        ]
        return [p for p in out if p.magic == self.magic] if magic_only else out

    async def loss_per_lot(self, symbol: str, side: Side, entry: float, sl: float) -> float | None:
        """Tiền lỗ (account currency, số dương) của 1.0 lot nếu chạm SL – dùng order_calc_profit."""
        otype = mt5.ORDER_TYPE_BUY if side is Side.BUY else mt5.ORDER_TYPE_SELL
        profit = await self._run(mt5.order_calc_profit, otype, symbol, 1.0, entry, sl)
        return abs(profit) if profit is not None else None

    async def margin_required(self, symbol: str, side: Side, volume: float, price: float) -> float | None:
        otype = mt5.ORDER_TYPE_BUY if side is Side.BUY else mt5.ORDER_TYPE_SELL
        return await self._run(mt5.order_calc_margin, otype, symbol, volume, price)

    # ================================================================== orders
    @staticmethod
    def filling_type(spec: SymbolSpec) -> int:
        """Chọn filling mode hợp lệ theo bitmask của symbol (FxPro thường là FOK/IOC)."""
        if spec.filling_mode & SYMBOL_FILLING_FOK:
            return mt5.ORDER_FILLING_FOK
        if spec.filling_mode & SYMBOL_FILLING_IOC:
            return mt5.ORDER_FILLING_IOC
        return mt5.ORDER_FILLING_RETURN

    def _send_sync(self, build_request: Callable[[], dict], label: str, dry_run: bool) -> OrderResult:
        """Gửi lệnh (hoặc order_check khi dry-run) có retry với các retcode tạm thời."""
        request: dict = {}
        for attempt in range(1, self.order_retries + 1):
            request = build_request()  # build lại để lấy giá mới nhất khi requote
            if dry_run:
                chk = mt5.order_check(request)
                if chk is None:
                    return OrderResult(False, -1, f"order_check None {self._last_error()}", True, request=request)
                ok = chk.retcode == 0
                return OrderResult(ok, chk.retcode, f"DRY-RUN order_check: {chk.comment}", True,
                                   price=request.get("price"), volume=request.get("volume"), request=request)
            res = mt5.order_send(request)
            if res is None:
                log.warning("%s: order_send None (%s) lần %d", label, self._last_error(), attempt)
                time.sleep(0.5 * attempt)
                continue
            if res.retcode in SUCCESS_RETCODES:
                return OrderResult(True, res.retcode, res.comment, False, ticket=res.order or res.deal,
                                   price=res.price, volume=res.volume, request=request)
            if res.retcode in RETRYABLE_RETCODES and attempt < self.order_retries:
                log.warning("%s: retcode %s (%s) – retry %d", label, res.retcode, res.comment, attempt)
                time.sleep(0.5 * attempt)
                continue
            return OrderResult(False, res.retcode, res.comment, False, request=request)
        return OrderResult(False, -1, f"order_send thất bại sau {self.order_retries} lần", False, request=request)

    async def open_position(
        self, spec: SymbolSpec, side: Side, volume: float, sl: float, tp: float | None, comment: str = "ai-bot"
    ) -> OrderResult:
        def build() -> dict:
            t = mt5.symbol_info_tick(spec.name)
            price = t.ask if side is Side.BUY else t.bid
            req = {
                "action": mt5.TRADE_ACTION_DEAL, "symbol": spec.name, "volume": float(volume),
                "type": mt5.ORDER_TYPE_BUY if side is Side.BUY else mt5.ORDER_TYPE_SELL,
                "price": price, "sl": float(sl), "deviation": self.deviation, "magic": self.magic,
                "comment": comment[:31], "type_time": mt5.ORDER_TIME_GTC, "type_filling": self.filling_type(spec),
            }
            if tp:
                req["tp"] = float(tp)
            return req

        return await self._run(self._send_sync, build, f"OPEN {side.value} {spec.name}", self.dry_run)

    async def modify_sl_tp(self, position: Position, sl: float | None, tp: float | None) -> OrderResult:
        def build() -> dict:
            return {"action": mt5.TRADE_ACTION_SLTP, "symbol": position.symbol, "position": position.ticket,
                    "sl": float(sl or 0.0), "tp": float(tp or 0.0), "magic": self.magic}

        return await self._run(self._send_sync, build, f"MODIFY #{position.ticket}", self.dry_run)

    async def close_position(self, position: Position, comment: str = "ai-bot close", force: bool = False) -> OrderResult:
        """Đóng 1 vị thế. `force=True` bỏ qua dry-run (dùng cho /close_all khẩn cấp)."""
        spec = await self.symbol_spec(position.symbol)

        def build() -> dict:
            t = mt5.symbol_info_tick(position.symbol)
            closing_side = position.side.opposite
            return {
                "action": mt5.TRADE_ACTION_DEAL, "symbol": position.symbol, "volume": position.volume,
                "type": mt5.ORDER_TYPE_BUY if closing_side is Side.BUY else mt5.ORDER_TYPE_SELL,
                "position": position.ticket, "price": t.ask if closing_side is Side.BUY else t.bid,
                "deviation": self.deviation, "magic": self.magic, "comment": comment[:31],
                "type_time": mt5.ORDER_TIME_GTC, "type_filling": self.filling_type(spec),
            }

        dry = self.dry_run and not force
        return await self._run(self._send_sync, build, f"CLOSE #{position.ticket}", dry)

    # ================================================================== history
    async def closed_trade_info(self, position: Position) -> ClosedTrade | None:
        """Tra lịch sử deal của 1 position đã đóng. None nếu lịch sử chưa đồng bộ."""
        deals = await self._run(lambda: mt5.history_deals_get(position=position.ticket))
        if not deals:
            return None
        exits = [d for d in deals if d.entry in (DEAL_ENTRY_OUT, DEAL_ENTRY_INOUT, DEAL_ENTRY_OUT_BY)]
        if not exits:
            return None
        net = sum(d.profit + d.swap + d.commission + getattr(d, "fee", 0.0) for d in deals)
        last = max(exits, key=lambda d: d.time_msc)
        acc = await self.account()
        return ClosedTrade(
            ticket=position.ticket, symbol=position.symbol, side=position.side.value,
            volume=sum(d.volume for d in exits), open_price=position.price_open, close_price=last.price,
            profit=round(net, 2), reason=DEAL_REASONS.get(last.reason, f"reason {last.reason}"),
            balance_after=acc.balance,
        )
