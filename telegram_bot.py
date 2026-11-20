"""Telegram Bot: thông báo realtime + điều khiển từ xa (python-telegram-bot v21+ async).

- Chỉ chấp nhận update từ `TELEGRAM_ADMIN_CHAT_ID`; mọi chat khác bị bỏ qua im lặng và ghi log.
- Tin nhắn đi được đưa vào hàng đợi, gửi tuần tự với khoảng cách tối thiểu để tránh rate-limit
  (xử lý `RetryAfter`, lỗi mạng tạm thời và cắt tin > 4096 ký tự).
"""
from __future__ import annotations

import asyncio
import html
import logging
import secrets
import time
from datetime import timedelta
from typing import Protocol

from telegram import BotCommand, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.error import NetworkError, RetryAfter, TelegramError, TimedOut
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes, MessageHandler, filters

log = logging.getLogger(__name__)

MAX_LEN = 4000
CONFIRM_TTL_SEC = 60


def esc(value: object) -> str:
    """Escape HTML cho parse_mode=HTML."""
    return html.escape(str(value), quote=False)


class BotController(Protocol):
    """Giao diện mà TradingBot cung cấp cho Telegram (tránh import vòng)."""

    @property
    def is_paused(self) -> bool: ...
    def pause(self) -> None: ...
    def resume(self) -> None: ...
    async def status_text(self) -> str: ...
    async def positions_text(self) -> str: ...
    async def stats_text(self, days: int | None = 30) -> str: ...
    async def close_all_positions(self) -> str: ...
    async def analyze_text(self, symbol: str) -> str: ...
    @property
    def scalp_enabled(self) -> bool: ...
    def set_scalp(self, enabled: bool) -> None: ...
    async def scalp_signal(self, symbol: str) -> str | None: ...
    async def execute_scalp(self, signal_id: str, choice: str) -> str: ...


class TelegramNotifier:
    """Quản lý Application của python-telegram-bot + hàng đợi gửi tin."""

    def __init__(self, token: str, admin_chat_id: int, min_interval_sec: float = 1.1) -> None:
        self.admin_chat_id = admin_chat_id
        self.min_interval = min_interval_sec
        self.controller: BotController | None = None
        self.app: Application = Application.builder().token(token).build()
        self._queue: asyncio.Queue[tuple[str, InlineKeyboardMarkup | None]] = asyncio.Queue(maxsize=500)
        self._sender_task: asyncio.Task | None = None
        self._polling = False
        self._pending_confirm: dict[str, float] = {}  # nonce -> expiry
        self._register_handlers()

    # ================================================================== lifecycle
    def attach(self, controller: BotController) -> None:
        self.controller = controller

    async def start(self, polling: bool = True) -> None:
        await self.app.initialize()
        self._sender_task = asyncio.create_task(self._sender_loop(), name="tg-sender")
        if polling:
            await self.app.start()
            await self.app.updater.start_polling(drop_pending_updates=True, allowed_updates=Update.ALL_TYPES)
            self._polling = True
            try:
                await self.app.bot.set_my_commands([
                    BotCommand("status", "Trạng thái bot & tài khoản"),
                    BotCommand("positions", "Vị thế đang mở"),
                    BotCommand("stats", "Winrate & P&L tích luỹ: /stats 30 hoặc /stats all"),
                    BotCommand("pause", "Tạm dừng mở lệnh mới"),
                    BotCommand("resume", "Tiếp tục mở lệnh mới"),
                    BotCommand("close_all", "Đóng toàn bộ vị thế (có xác nhận)"),
                    BotCommand("analyze", "Phân tích AI ngay: /analyze GOLD"),
                    BotCommand("scalp", "Lướt sóng: /scalp on|off hoặc /scalp GOLD"),
                    BotCommand("help", "Hướng dẫn"),
                ])
            except TelegramError as exc:
                log.warning("set_my_commands lỗi: %s", exc)
        log.info("Telegram started (polling=%s)", polling)

    async def stop(self, flush_timeout: float = 10.0) -> None:
        if self._sender_task is None:  # chưa từng start
            return
        await self.flush(flush_timeout)
        if self._sender_task:
            self._sender_task.cancel()
            await asyncio.gather(self._sender_task, return_exceptions=True)
        if self._polling:
            await self.app.updater.stop()
            await self.app.stop()
        await self.app.shutdown()

    async def flush(self, timeout: float = 10.0) -> None:
        """Đợi hàng đợi gửi hết (dùng khi --once hoặc shutdown)."""
        try:
            await asyncio.wait_for(self._queue.join(), timeout)
        except asyncio.TimeoutError:
            log.warning("Telegram flush timeout – %d tin chưa gửi", self._queue.qsize())

    # ================================================================== outgoing
    def notify(self, text: str, keyboard: InlineKeyboardMarkup | None = None) -> None:
        """Đưa tin vào hàng đợi (không chặn). Text là HTML đã escape."""
        chunks = [text[i : i + MAX_LEN] for i in range(0, len(text), MAX_LEN)] or [""]
        for i, chunk in enumerate(chunks):
            try:
                self._queue.put_nowait((chunk, keyboard if i == len(chunks) - 1 else None))
            except asyncio.QueueFull:
                log.error("Telegram queue đầy – bỏ tin: %s", chunk[:80])

    def notify_scalp(self, text: str, signal_id: str, ai_side: str | None) -> None:
        """Tín hiệu lướt sóng kèm nút LONG / SHORT / Bỏ qua (⭐ = hướng AI gợi ý)."""
        star = {"BUY": ("⭐ ", ""), "SELL": ("", "⭐ ")}.get(ai_side or "", ("", ""))
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton(f"{star[0]}🟢 LONG", callback_data=f"scalp:BUY:{signal_id}"),
             InlineKeyboardButton(f"{star[1]}🔴 SHORT", callback_data=f"scalp:SELL:{signal_id}")],
            [InlineKeyboardButton("❌ Bỏ qua", callback_data=f"scalp:SKIP:{signal_id}")],
        ])
        self.notify(text, kb)

    async def _sender_loop(self) -> None:
        last_sent = 0.0
        while True:
            text, kb = await self._queue.get()
            try:
                wait = self.min_interval - (time.monotonic() - last_sent)
                if wait > 0:
                    await asyncio.sleep(wait)
                await self._send_with_retry(text, kb)
                last_sent = time.monotonic()
            except Exception as exc:  # noqa: BLE001 - không để sender chết
                log.error("Gửi Telegram thất bại: %s", exc)
            finally:
                self._queue.task_done()

    async def _send_with_retry(self, text: str, kb: InlineKeyboardMarkup | None, attempts: int = 4) -> None:
        for attempt in range(1, attempts + 1):
            try:
                await self.app.bot.send_message(
                    chat_id=self.admin_chat_id, text=text, parse_mode=ParseMode.HTML,
                    reply_markup=kb, disable_web_page_preview=True,
                )
                return
            except RetryAfter as exc:
                ra = exc.retry_after
                delay = ra.total_seconds() if isinstance(ra, timedelta) else float(ra)
                log.warning("Telegram RetryAfter %.1fs", delay)
                await asyncio.sleep(delay + 0.5)
            except (TimedOut, NetworkError) as exc:
                log.warning("Telegram network lỗi (lần %d): %s", attempt, exc)
                await asyncio.sleep(2 ** attempt)
        raise TelegramError(f"Không gửi được sau {attempts} lần")

    # ================================================================== incoming
    def _register_handlers(self) -> None:
        admin = filters.Chat(chat_id=self.admin_chat_id)
        commands = {
            "start": self._cmd_help, "help": self._cmd_help, "status": self._cmd_status,
            "positions": self._cmd_positions, "stats": self._cmd_stats, "pause": self._cmd_pause, "resume": self._cmd_resume,
            "close_all": self._cmd_close_all, "analyze": self._cmd_analyze, "scalp": self._cmd_scalp,
        }
        for name, handler in commands.items():
            self.app.add_handler(CommandHandler(name, handler, filters=admin))
        self.app.add_handler(CallbackQueryHandler(self._on_callback, pattern=r"^closeall:"))
        self.app.add_handler(CallbackQueryHandler(self._on_scalp_callback, pattern=r"^scalp:"))
        # Mọi thứ không phải từ admin -> log và bỏ qua (không trả lời để không lộ bot)
        self.app.add_handler(MessageHandler(~admin, self._on_unauthorized), group=1)
        self.app.add_error_handler(self._on_error)

    def _is_admin(self, update: Update) -> bool:
        chat = update.effective_chat
        return chat is not None and chat.id == self.admin_chat_id

    async def _on_unauthorized(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        user = update.effective_user
        log.warning("Bỏ qua tin nhắn từ chat lạ: chat_id=%s user=%s",
                    update.effective_chat.id if update.effective_chat else "?",
                    f"{user.id}/@{user.username}" if user else "?")

    async def _on_error(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        log.error("Telegram handler lỗi: %s", context.error, exc_info=context.error)

    async def _reply(self, update: Update, text: str, kb: InlineKeyboardMarkup | None = None) -> None:
        if update.effective_message:
            for i in range(0, max(len(text), 1), MAX_LEN):
                await update.effective_message.reply_text(
                    text[i : i + MAX_LEN], parse_mode=ParseMode.HTML, reply_markup=kb, disable_web_page_preview=True
                )

    def _ctrl(self) -> BotController:
        if self.controller is None:
            raise RuntimeError("Telegram chưa được gắn controller")
        return self.controller

    async def _cmd_help(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        await self._reply(update, (
            "<b>🤖 MT5 AI Bot</b>\n"
            "/status – trạng thái bot, balance, equity, P&amp;L hôm nay\n"
            "/positions – vị thế đang mở\n"
            "/stats [ngày|all] – winrate, P&amp;L tích luỹ (mặc định 30 ngày)\n"
            "/pause – tạm dừng mở lệnh mới\n"
            "/resume – tiếp tục\n"
            "/close_all – đóng toàn bộ vị thế (có xác nhận)\n"
            "/analyze &lt;SYMBOL&gt; – phân tích AI ngay (không vào lệnh)\n"
            "/scalp on|off – bật/tắt tín hiệu lướt sóng khung ngắn (có nút LONG/SHORT/Bỏ qua)\n"
            "/scalp &lt;SYMBOL&gt; – lấy tín hiệu lướt sóng ngay"
        ))

    async def _cmd_status(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        await self._reply(update, await self._ctrl().status_text())

    async def _cmd_positions(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        await self._reply(update, await self._ctrl().positions_text())

    async def _cmd_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        arg = (context.args or ["30"])[0].strip().lower()
        if arg == "all":
            days: int | None = None
        elif arg.isdigit() and 1 <= int(arg) <= 3650:
            days = int(arg)
        else:
            await self._reply(update, "Cú pháp: <code>/stats</code>, <code>/stats 7</code> hoặc <code>/stats all</code>")
            return
        await self._reply(update, await self._ctrl().stats_text(days))

    async def _cmd_pause(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        self._ctrl().pause()
        await self._reply(update, "⏸ Đã <b>TẠM DỪNG</b> mở lệnh mới. Lệnh đang mở vẫn được giám sát.")

    async def _cmd_resume(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        self._ctrl().resume()
        await self._reply(update, "▶️ Đã <b>TIẾP TỤC</b> cho phép AI mở lệnh mới.")

    async def _cmd_close_all(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        now = time.monotonic()
        self._pending_confirm = {k: v for k, v in self._pending_confirm.items() if v > now}
        nonce = secrets.token_hex(4)
        self._pending_confirm[nonce] = now + CONFIRM_TTL_SEC
        kb = InlineKeyboardMarkup([[
            InlineKeyboardButton("✅ Confirm", callback_data=f"closeall:yes:{nonce}"),
            InlineKeyboardButton("❌ Cancel", callback_data=f"closeall:no:{nonce}"),
        ]])
        await self._reply(update, f"⚠️ <b>Đóng TOÀN BỘ vị thế?</b>\nXác nhận trong {CONFIRM_TTL_SEC}s.", kb)

    async def _on_callback(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        if query is None:
            return
        if not self._is_admin(update):
            await query.answer("Không có quyền.", show_alert=True)
            log.warning("Callback từ chat lạ bị từ chối: %s", update.effective_chat)
            return
        _, choice, nonce = (query.data or "::").split(":", 2)
        expiry = self._pending_confirm.pop(nonce, None)
        await query.answer()
        if expiry is None or expiry < time.monotonic():
            await query.edit_message_text("⌛ Yêu cầu đã hết hạn hoặc đã xử lý. Gõ lại /close_all.")
            return
        if choice != "yes":
            await query.edit_message_text("❎ Đã huỷ lệnh đóng toàn bộ.")
            return
        await query.edit_message_text("⏳ Đang đóng toàn bộ vị thế...")
        result = await self._ctrl().close_all_positions()
        await query.edit_message_text(result[:MAX_LEN], parse_mode=ParseMode.HTML)

    async def _cmd_analyze(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        args = context.args or []
        symbol = args[0].strip() if args else ""
        if not symbol or len(symbol) > 32 or not all(c.isalnum() or c in "._#-" for c in symbol):
            await self._reply(update, "Cú pháp: <code>/analyze GOLD</code>")
            return
        await self._reply(update, f"🔎 Đang phân tích <b>{esc(symbol)}</b>...")
        await self._reply(update, await self._ctrl().analyze_text(symbol))

    async def _cmd_scalp(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        ctrl = self._ctrl()
        arg = (context.args or [""])[0].strip()
        if arg.lower() in ("on", "off"):
            ctrl.set_scalp(arg.lower() == "on")
            await self._reply(update, "⚡ Lướt sóng: <b>BẬT</b> – bot sẽ gửi tín hiệu kèm nút mỗi nến mới."
                              if arg.lower() == "on" else "⚡ Lướt sóng: <b>TẮT</b>.")
            return
        if not arg:
            state = "BẬT" if ctrl.scalp_enabled else "TẮT"
            await self._reply(update, f"⚡ Lướt sóng đang <b>{state}</b>.\n"
                              "Cú pháp: <code>/scalp on</code>, <code>/scalp off</code>, <code>/scalp GOLD</code>")
            return
        if len(arg) > 32 or not all(c.isalnum() or c in "._#-" for c in arg):
            await self._reply(update, "Cú pháp: <code>/scalp GOLD</code>")
            return
        await self._reply(update, f"⚡ Đang phân tích lướt sóng <b>{esc(arg)}</b>...")
        try:
            msg = await ctrl.scalp_signal(arg)
        except Exception as exc:  # noqa: BLE001
            msg = f"❌ {esc(exc)}"
        if msg:
            await self._reply(update, msg)

    async def _on_scalp_callback(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        if query is None:
            return
        if not self._is_admin(update):
            await query.answer("Không có quyền.", show_alert=True)
            log.warning("Scalp callback từ chat lạ bị từ chối: %s", update.effective_chat)
            return
        _, choice, signal_id = (query.data or "::").split(":", 2)
        await query.answer()
        original = query.message.text_html if query.message and query.message.text else ""
        if choice != "SKIP":
            await query.edit_message_text(f"{original}\n\n⏳ Đang vào lệnh {choice}...", parse_mode=ParseMode.HTML)
        try:
            result = await self._ctrl().execute_scalp(signal_id, choice)
        except Exception as exc:  # noqa: BLE001
            log.exception("execute_scalp lỗi")
            result = f"❌ Lỗi: {esc(exc)}"
        await query.edit_message_text(f"{original}\n\n{result}"[-MAX_LEN:], parse_mode=ParseMode.HTML)


class NullNotifier:
    """Dùng khi không cấu hình Telegram – chỉ log ra console."""

    def attach(self, controller: BotController) -> None:  # noqa: D401
        pass

    async def start(self, polling: bool = True) -> None:
        log.info("Telegram disabled (thiếu TELEGRAM_BOT_TOKEN / TELEGRAM_ADMIN_CHAT_ID)")

    async def stop(self, flush_timeout: float = 0) -> None:
        pass

    async def flush(self, timeout: float = 0) -> None:
        pass

    def notify(self, text: str, keyboard: object = None) -> None:
        log.info("[notify] %s", html.unescape(text).replace("<b>", "").replace("</b>", ""))

    def notify_scalp(self, text: str, signal_id: str, ai_side: str | None) -> None:
        self.notify(text + "\n(Telegram tắt – không có nút bấm)")
