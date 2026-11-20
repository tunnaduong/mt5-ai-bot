# MT5 AI Trading Bot (FxPro) + Telegram

Bot giao dịch tự động: **MetaTrader 5 (FxPro)** ↔ **AI endpoint (LLM/HTTP)** ↔ **Risk Manager** ↔ **Telegram**,
chạy trên một event loop `asyncio`, mặc định ở chế độ **DRY-RUN** (không gửi lệnh thật).

> ⚠️ Giao dịch CFD/Forex có rủi ro mất vốn cao. Hãy chạy DRY-RUN và tài khoản DEMO ít nhất vài tuần
> trước khi cân nhắc tiền thật. Phần mềm không đảm bảo lợi nhuận.

---

## 1. Sơ đồ luồng dữ liệu

```
                          ┌───────────────────────────────────────────┐
                          │                 bot.py                     │
                          │  TradingBot (asyncio, 2 vòng lặp song song)│
                          │   • trading loop  (mỗi LOOP_INTERVAL_SEC)  │
                          │   • monitor loop  (mỗi MONITOR_INTERVAL_SEC)│
                          └───┬───────────┬────────────┬──────────┬───┘
                              │           │            │          │
        ① nến mới?            │           │            │          │ ⑥ log mọi bước
   ┌──────────────────────────▼─┐   ┌─────▼────────┐   │   ┌──────▼───────┐
   │ mt5_client.py (1 thread)   │   │ indicators.py│   │   │ storage.py   │
   │ tick/bars/account/positions│──►│ EMA RSI ATR  │   │   │ SQLite +     │
   │ order_send / order_check   │   │ nến, swing   │   │   │ logs/bot.log │
   └───────────▲────────────────┘   └─────┬────────┘   │   └──────────────┘
               │                          │ ② snapshot JSON gọn
               │                    ┌─────▼────────┐
               │                    │ ai_client.py │ httpx + retry/backoff
               │                    │ → AI endpoint│ Pydantic validate
               │                    └─────┬────────┘ lỗi → HOLD
               │                          │ ③ {action, confidence, SL, TP, reasoning}
               │                    ┌─────▼──────────┐
               │  ⑤ order_send     │ risk_manager.py │ spread, daily loss, số lệnh,
               └────────────────────┤  (bắt buộc)    │ SL/TP/stops_level, R:R, lot
                    (nếu APPROVED) └─────┬──────────┘ theo % equity, margin
                                          │ ④ thông báo / risk block
                                    ┌─────▼──────────┐         ┌───────────┐
                                    │ telegram_bot.py│◄───────►│ Admin chat│
                                    │ queue chống    │ lệnh    │ (whitelist)│
                                    │ rate-limit     │ /status │           │
                                    └────────────────┘ ...     └───────────┘
```

**Monitor loop** (độc lập với AI): kiểm tra kết nối MT5 (tự reconnect + cảnh báo), phát hiện lệnh đóng
(SL/TP/manual/stop-out → tra `history_deals_get` để lấy P&L thật), daily loss limit, sức khoẻ AI,
và gửi báo cáo cuối ngày.

## 2. Cấu trúc thư mục

```
mt5-ai-bot/
├── bot.py              # entrypoint: TradingBot, vòng lặp, CLI (--once, --dry-run), graceful shutdown
├── config.py           # Settings (pydantic-settings) đọc .env
├── models.py           # dataclass trung lập: SymbolSpec, Tick, AccountState, Position, ...
├── mt5_client.py       # wrapper MetaTrader5 async (executor 1 luồng), retry, filling mode động
├── indicators.py       # EMA, RSI, ATR, mẫu nến, summarize() cho AI
├── ai_client.py        # gọi AI (OpenAI-compatible / custom), parse + validate, fallback HOLD
├── risk_manager.py     # kiểm soát rủi ro + tính lot
├── telegram_bot.py     # command, inline confirm, hàng đợi gửi tin
├── storage.py          # SQLite: ai_calls, risk_checks, orders, closed_trades, events, kv
├── tests/              # pytest: risk manager, AI client, luồng bot (mock MT5 + mock AI)
├── .env.example
├── requirements.txt / requirements-dev.txt
└── pytest.ini
```

## 3. Cài đặt (Windows – bắt buộc cho MT5)

Package `MetaTrader5` chỉ chạy trên **Windows** (giao tiếp trực tiếp với terminal MT5 đang mở).
Test unit thì chạy được trên mọi OS.

1. Cài **FxPro MetaTrader 5** và đăng nhập tài khoản **Demo** (FxPro Direct → Open Demo Account → MT5).
2. Trong terminal: `Tools → Options → Expert Advisors` → tick **Allow algorithmic trading**, rồi bật nút
   **Algo Trading** trên toolbar (màu xanh).
3. Mở `Market Watch` (Ctrl+M) → chuột phải → *Show All* để chắc các symbol (`GOLD`, `EURUSD`,
   `BITCOIN`, `ANT.O`...) tồn tại. Tên symbol phải khớp **chính xác** với broker.
4. Python ≥ 3.11:
   ```powershell
   cd mt5-ai-bot
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements-dev.txt
   copy .env.example .env      # rồi sửa .env
   ```

## 4. Tạo Telegram bot và lấy Chat ID

1. Mở Telegram, chat với **@BotFather** → gửi `/newbot`.
2. Đặt tên hiển thị, rồi username kết thúc bằng `bot` (vd. `my_mt5_ai_bot`).
3. BotFather trả về token dạng `123456789:AAH...` → dán vào `TELEGRAM_BOT_TOKEN`. **Không chia sẻ token.**
4. Mở chat với bot vừa tạo, bấm **Start** (gửi `/start`) – bắt buộc để bot được phép nhắn cho bạn.
5. Lấy Chat ID (chọn 1 cách):
   - Chat với **@userinfobot** → nó trả về `Id: 123456789`; hoặc
   - Mở trình duyệt: `https://api.telegram.org/bot<TOKEN>/getUpdates` → tìm `"chat":{"id":123456789,...}`.
6. Điền `TELEGRAM_ADMIN_CHAT_ID=123456789`. Bot chỉ phản hồi chat này; mọi người khác bị bỏ qua và ghi log.
7. (Tuỳ chọn) BotFather → `/setjoingroups` → Disable để không ai thêm bot vào nhóm.

## 5. Cấu hình AI endpoint

| Nhà cung cấp | `AI_ENDPOINT_URL` | Ghi chú |
|---|---|---|
| OpenAI | `https://api.openai.com/v1/chat/completions` | `AI_JSON_MODE=true` |
| OpenRouter | `https://openrouter.ai/api/v1/chat/completions` | model vd. `anthropic/claude-sonnet-4.5` |
| Ollama local | `http://localhost:11434/v1/chat/completions` | `AI_API_KEY` để trống |
| Endpoint riêng | URL của bạn | `AI_API_STYLE=custom` |

Nếu endpoint không hỗ trợ `response_format`, đặt `AI_JSON_MODE=false` – parser vẫn bóc được JSON
kể cả khi model bọc trong ```` ```json ````.

**Custom endpoint** nhận `POST {"model":..., "system": "<prompt>", "snapshot": {...}}` và trả về
```json
{"action": "BUY", "confidence": 0.78, "stop_loss": 2391.5, "take_profit": 2421.0, "reasoning": "..."}
```
(hoặc bọc trong `{"decision": {...}}`).

## 6. Chạy thử trên tài khoản Demo – từng bước

```powershell
# 0. Unit test (không cần MT5)
pytest -q

# 1. Một nhịp phân tích, không Telegram, không gửi lệnh – kiểm tra kết nối MT5 + AI
python bot.py --once --dry-run --no-telegram --log-level DEBUG

# 2. Một nhịp có Telegram (nhận tin "Bot khởi động", tín hiệu, risk block)
python bot.py --once --dry-run

# 3. Chạy liên tục ở DRY-RUN, thử /status /positions /analyze GOLD /pause /resume trên Telegram
python bot.py --dry-run

# 4. Khi hài lòng: DRY_RUN=false trong .env, VẪN trên tài khoản DEMO
python bot.py
```

Ở **DRY-RUN**, bot vẫn gọi `mt5.order_check()` với request thật → bạn biết lệnh có hợp lệ với broker
(filling mode, volume, stops level, margin) mà không khớp lệnh. `/close_all` ở DRY-RUN cũng chỉ mô phỏng.

**Lớp bảo vệ tài khoản thật:** nếu `DRY_RUN=false` và tài khoản là REAL, bot từ chối chạy trừ khi
`ALLOW_REAL_ACCOUNT=true`.

Dừng bot: `Ctrl+C` – bot gửi nốt tin Telegram trong hàng đợi, đóng HTTP client, `mt5.shutdown()`.
**Lệnh đang mở KHÔNG bị đóng khi tắt bot** (vẫn có SL/TP trên server).

## 7. Lệnh Telegram

| Lệnh | Chức năng |
|---|---|
| `/status` | Chạy/tạm dừng, Live/Dry-run, Balance, Equity, Margin level, P&L hôm nay |
| `/positions` | Vị thế đang mở + P&L (đánh dấu `bot` / `manual`) |
| `/stats [ngày\|all]` | Winrate, P&L, profit factor, max drawdown, chuỗi thua – theo mã và theo ngày (mặc định 30 ngày) |
| `/pause`, `/resume` | Dừng/tiếp tục **mở lệnh mới** (monitor SL/TP vẫn chạy; khi pause bot không gọi AI) |
| `/close_all` | Đóng mọi vị thế, cần bấm **Confirm** trong 60 giây |
| `/analyze GOLD` | Chạy AI ngay + chỉ báo + risk check thử – **không vào lệnh** |

Thông báo tự động: vào lệnh, đóng lệnh (lý do SL/TP/manual/stop-out, P&L $, % balance), risk block
(khi confidence ≥ `NOTIFY_MIN_CONFIDENCE`), mất/có lại kết nối MT5, AI lỗi liên tiếp, daily loss limit,
tổng kết ngày lúc `DAILY_SUMMARY_HOUR_UTC`.

## 8. Risk Manager – các lớp kiểm tra

1. Action phải là BUY/SELL, `confidence ≥ MIN_CONFIDENCE`.
2. `trade_mode` của symbol (disabled / close-only / long-only / short-only) và quyền trade của tài khoản.
3. Tick còn mới (`MAX_TICK_AGE_SEC`, đo bằng đồng hồ local để tránh lệch múi giờ server FxPro).
4. Spread ≤ `MAX_SPREAD_POINTS` (override theo symbol) **và** ≤ `MAX_SPREAD_ATR_RATIO × ATR`.
5. Daily loss: `(equity − balance đầu ngày UTC) / balance đầu ngày ≤ −DAILY_MAX_LOSS_PCT` → chặn.
6. Số lệnh mở toàn tài khoản / theo symbol.
7. SL bắt buộc, đúng phía, cách giá ≥ `max(trade_stops_level, trade_freeze_level) × point`,
   trong khoảng `[MIN_SL_ATR_MULT, MAX_SL_ATR_MULT] × ATR`.
8. TP đúng phía, R:R ≥ `MIN_RR`.
9. Lot = `equity × RISK_PER_TRADE_PCT% / lỗ_mỗi_lot`. Lỗ mỗi lot lấy từ `mt5.order_calc_profit`
   (chính xác cho cặp chéo, vàng, crypto, cổ phiếu); fallback `tick_value/tick_size`. Làm tròn **xuống**
   theo `volume_step`, kẹp `volume_max`; nếu < `volume_min` → **chặn**, trừ khi bật `MIN_LOT_MAX_RISK_PCT`
   và rủi ro của lot tối thiểu nằm dưới trần đó (xem mục 11).
10. Margin yêu cầu (`order_calc_margin`) ≤ `MAX_MARGIN_USAGE_PCT` × free margin.

Mọi thuộc tính symbol (`digits`, `point`, `volume_*`, `filling_mode`, `trade_stops_level`...) được đọc
động từ `symbol_info` mỗi lần – không hardcode. Filling mode chọn theo bitmask: FOK → IOC → RETURN.

## 9. Debug & dữ liệu

- `logs/bot.log` (xoay vòng 5MB × 10).
- `data/bot.db` (SQLite) – mở bằng DB Browser for SQLite:
  - `ai_calls`: snapshot gửi đi, raw response, decision, latency, lỗi
  - `risk_checks`: từng quyết định + lý do chặn
  - `orders`: request gửi MT5, retcode, dry-run hay thật
  - `closed_trades`: P&L, lý do đóng (dùng cho Daily Summary)
  - `events`: mất kết nối, pause/resume, daily loss...

Ví dụ: `SELECT ts, symbol, action, approved, reasons FROM risk_checks ORDER BY id DESC LIMIT 20;`

## 10. Lưu ý FxPro

- Symbol FxPro: `GOLD` (không phải XAUUSD), `GOLDgr` (vàng theo gram), `BITCOIN`, cổ phiếu Mỹ dạng `ANT.O`. Crypto/cổ phiếu có
  spread lớn và giờ giao dịch riêng → đặt `MAX_SPREAD_OVERRIDES` phù hợp, xem spread thực tế bằng `/analyze`.
- Tài khoản FxPro MT5 là market execution → filling FOK/IOC; bot tự chọn theo symbol.
- Cổ phiếu thường có `volume_min` lớn (vd. 1 share/lot) – với vốn nhỏ, risk check có thể chặn vì
  "lot < volume_min": đây là hành vi đúng.
- Thời gian server FxPro là GMT+2/+3; bot dùng **UTC** cho mốc ngày (daily loss, summary).
- Lệnh đóng chỉ được ghi nhận/thông báo khi bot đang chạy (bot so sánh danh sách vị thế giữa các nhịp).

## 11. Tài khoản nhỏ (~$100)

Với $100, rủi ro 0.5–1%/lệnh chỉ là $0.5–1, nhỏ hơn mức lỗ của **lot tối thiểu** ở mọi SL hợp lý
→ mặc định bot chặn hết lệnh. `MIN_LOT_MAX_RISK_PCT` cho phép dùng lot tối thiểu nếu rủi ro thực của nó
không vượt trần; bot cũng gửi cho AI `max_affordable_sl_distance` để AI chọn SL vừa túi (hoặc HOLD).

Lỗ của lot tối thiểu trên FxPro (tài khoản USD):

| Symbol | Lot tối thiểu | Lỗ mỗi $1 giá | SL tối đa với trần 5% của $100 |
|---|---|---|---|
| `BITCOIN` | 0.01 BTC | $0.01 | $500 (broker yêu cầu SL ≥ $200) |
| `GOLDgr` | 5 gram | $5 / $1 mỗi gram (≈ $0.16 mỗi $1/oz) | $1/gram ≈ $31/oz |
| `GOLD` | 0.01 lot = 1 oz | $1 | $5 → quá sát, gần như luôn bị chặn |

Cấu hình gợi ý (chấp nhận rủi ro cao hơn nhiều so với tài khoản lớn):

```ini
SYMBOLS=BITCOIN,GOLDgr          # dùng GOLDgr thay cho GOLD
RISK_PER_TRADE_PCT=1
MIN_LOT_MAX_RISK_PCT=5           # mỗi lệnh lỗ tối đa ~$5
MAX_OPEN_POSITIONS=1             # 1 lệnh tại một thời điểm
DAILY_MAX_LOSS_PCT=10            # 2 lệnh thua liên tiếp là dừng trong ngày
MAX_SPREAD_OVERRIDES=BITCOIN:5000,GOLDgr:30
MAX_MARGIN_USAGE_PCT=60          # đòn bẩy tài khoản thật thấp hơn demo
```

Lưu ý: 10 lệnh thua liên tiếp ở trần 5% ≈ mất 40% tài khoản. Hãy kiểm chứng bằng `/stats` trên
demo trước (có thể mở 1 tài khoản demo $100 để thử đúng cấu hình này).
