"""
هشدار RCI با فرمان تلگرام - BTCUSDT
------------------------------------
کدهایی که به ربات می‌فرستید:
    00 -> لغو همه‌ی هشدارهای فعال
    10 -> هشدار وقتی RCI81 در تایم‌فریم ۱ دقیقه به -۹۰ برسد (عبور نزولی)
    11 -> هشدار وقتی RCI81 در تایم‌فریم ۱ دقیقه به +۹۰ برسد (عبور صعودی)
    50 -> هشدار وقتی RCI81 در تایم‌فریم ۵ دقیقه به -۸۰ برسد (عبور نزولی)
    51 -> هشدار وقتی RCI81 در تایم‌فریم ۵ دقیقه به +۸۰ برسد (عبور صعودی)

- کدها با هم قابل ترکیب‌اند (مثلاً هم ۱۰ و هم ۵۱ فعال باشد).
- هر کد تا وقتی ۰۰ نفرستید فعال می‌ماند و هر بار که عبور رخ دهد پیام می‌فرستد.
- فقط پیام‌های چتِ TELEGRAM_CHAT_ID پذیرفته می‌شود.

وضعیت (کدهای فعال + آخرین کندل بررسی‌شده + offset تلگرام) در فایل state.json
نگه داشته می‌شود؛ workflow این فایل را بین اجراها در یک برنچ جدا ذخیره می‌کند.
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import requests

SYMBOL = "BTCUSDT"
RCI_LENGTH = 81
STATE_FILE = "state.json"
BASE_URL = "https://data-api.binance.vision/api/v3/klines"
KLINE_LIMIT = 200
MAX_SCAN = 10  # اگر اجرایی دیر شد، حداکثر این‌قدر کندل اخیر برای عبورهای از قلم‌افتاده بررسی می‌شود

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

# direction: "down" یعنی عبور از بالا به پایین‌تر از level ، "up" یعنی عبور از پایین به بالاتر
COMMANDS = {
    "10": {"tf": "1m", "direction": "down", "level": -90},
    "11": {"tf": "1m", "direction": "up", "level": 90},
    "50": {"tf": "5m", "direction": "down", "level": -80},
    "51": {"tf": "5m", "direction": "up", "level": 80},
}

DIGIT_MAP = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")

HELP_TEXT = (
    "کدهای قابل استفاده:\n"
    "00 → لغو همه‌ی هشدارها\n"
    "10 → RCI در ۱ دقیقه به -90 برسد\n"
    "11 → RCI در ۱ دقیقه به +90 برسد\n"
    "50 → RCI در ۵ دقیقه به -80 برسد\n"
    "51 → RCI در ۵ دقیقه به +80 برسد"
)


# ---------------------------------------------------------------- تلگرام
def tg_api(method: str, **params) -> dict:
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/{method}"
    resp = requests.post(url, data=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def send_telegram(text: str) -> None:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ توکن یا چت‌آیدی تنظیم نشده؛ پیام ارسال نشد:\n" + text)
        return
    tg_api("sendMessage", chat_id=TELEGRAM_CHAT_ID, text=text)


# ---------------------------------------------------------------- دیتا و RCI
def fetch_klines(interval: str) -> pd.DataFrame:
    params = {"symbol": SYMBOL, "interval": interval, "limit": KLINE_LIMIT}
    resp = requests.get(BASE_URL, params=params, timeout=10)
    if resp.status_code != 200:
        print(f"خطا در واکشی {interval}: {resp.status_code} {resp.text[:300]}")
    resp.raise_for_status()
    df = pd.DataFrame(
        resp.json(),
        columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "qav", "trades", "tb_base", "tb_quote", "ignore",
        ],
    )
    df["close"] = df["close"].astype(float)
    df["open_time"] = df["open_time"].astype("int64")
    # آخرین کندل هنوز در حال تشکیل است؛ حذفش می‌کنیم
    return df.iloc[:-1].reset_index(drop=True)


def calc_rci(series: pd.Series, length: int) -> pd.Series:
    n = len(series)
    values = np.full(n, np.nan)
    prices = series.values
    for end in range(length - 1, n):
        window = prices[end - length + 1: end + 1]
        date_rank = np.arange(length)
        price_rank = pd.Series(window).rank(method="first").values - 1
        d = date_rank - price_rank
        values[end] = (1 - (6 * np.sum(d ** 2)) / (length * (length ** 2 - 1))) * 100
    return pd.Series(values, index=series.index)


def fmt_time(ms: int, interval_minutes: int) -> str:
    """زمان بسته‌شدن کندل به وقت تهران."""
    ts = pd.to_datetime(ms, unit="ms", utc=True) + pd.Timedelta(minutes=interval_minutes)
    try:
        ts = ts.tz_convert("Asia/Tehran")
        return ts.strftime("%Y-%m-%d %H:%M") + " (تهران)"
    except Exception:
        return ts.strftime("%Y-%m-%d %H:%M") + " UTC"


# ---------------------------------------------------------------- وضعیت
def load_state() -> dict:
    try:
        with open(STATE_FILE, encoding="utf-8") as f:
            state = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        state = {}
    state.setdefault("active", {})
    return state


def save_state(state: dict) -> None:
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2, sort_keys=True)


# ---------------------------------------------------------------- فرمان‌ها
def describe(code: str) -> str:
    c = COMMANDS[code]
    tf = "۱ دقیقه" if c["tf"] == "1m" else "۵ دقیقه"
    way = "از بالا به پایین" if c["direction"] == "down" else "از پایین به بالا"
    return f"RCI81 در تایم‌فریم {tf} {way} به {c['level']:+d} برسد"


def process_commands(state: dict, klines_cache: dict) -> None:
    params = {"timeout": 0, "allowed_updates": json.dumps(["message"])}
    first_run = "offset" not in state
    if not first_run:
        params["offset"] = state["offset"]

    updates = tg_api("getUpdates", **params).get("result", [])
    for u in updates:
        state["offset"] = u["update_id"] + 1
        msg = u.get("message")
        if not msg or str(msg["chat"]["id"]) != str(TELEGRAM_CHAT_ID):
            continue
        # در اولین اجرا پیام‌های قدیمی (مثل /start قبلی) نادیده گرفته می‌شوند
        if first_run and time.time() - msg.get("date", 0) > 900:
            continue

        text = (msg.get("text") or "").strip().translate(DIGIT_MAP)

        if text == "00":
            had = len(state["active"])
            state["active"] = {}
            send_telegram("✅ همه‌ی هشدارها لغو شد." if had else "هیچ هشدار فعالی نبود؛ همه‌چیز خاموش است.")
        elif text in COMMANDS:
            tf = COMMANDS[text]["tf"]
            if tf not in klines_cache:
                klines_cache[tf] = fetch_klines(tf)
            # از آخرین کندل بسته‌شده شروع می‌کنیم تا هشدار قدیمی نیاید
            state["active"][text] = int(klines_cache[tf]["open_time"].iloc[-1])
            active_list = "\n".join(f"• {k}: {describe(k)}" for k in sorted(state["active"]))
            send_telegram(f"✅ کد {text} فعال شد:\n{describe(text)}\n\nهشدارهای فعال الان:\n{active_list}")
        else:
            send_telegram(HELP_TEXT)

    if first_run and "offset" not in state:
        state["offset"] = 0  # تا وضعیت «اجرای اول» تکرار نشود


# ---------------------------------------------------------------- هشدارها
def check_alerts(state: dict, klines_cache: dict) -> None:
    rci_cache = {}
    for code in sorted(state["active"]):
        c = COMMANDS[code]
        tf = c["tf"]
        if tf not in klines_cache:
            klines_cache[tf] = fetch_klines(tf)
        df = klines_cache[tf]
        if tf not in rci_cache:
            rci_cache[tf] = calc_rci(df["close"], RCI_LENGTH)
        rci = rci_cache[tf]

        last_checked = state["active"][code]
        minutes = 1 if tf == "1m" else 5
        start = max(1, len(df) - MAX_SCAN)

        for i in range(start, len(df)):
            if df["open_time"].iloc[i] <= last_checked:
                continue
            prev, curr = rci.iloc[i - 1], rci.iloc[i]
            if pd.isna(prev) or pd.isna(curr):
                continue
            level = c["level"]
            if c["direction"] == "down":
                hit = prev > level >= curr
                icon, way = "🔴🐦‍🔥🍁", "نزولی"
            else:
                hit = prev < level <= curr
                icon, way = "🟢🐉🐲", "صعودی"
            if hit:
                send_telegram(
                    f"{icon} {SYMBOL} — RCI81 ({tf}) به {level:+d} رسید (عبور {way})\n"
                    f"مقدار فعلی: {curr:.2f} | قبلی: {prev:.2f}\n"
                    f"زمان کندل: {fmt_time(int(df['open_time'].iloc[i]), minutes)}\n"
                    f"(کد {code} — برای لغو: 00)"
                )

        state["active"][code] = int(df["open_time"].iloc[-1])


def main() -> int:
    state = load_state()
    klines_cache: dict = {}

    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        process_commands(state, klines_cache)
    else:
        print("⚠️ TELEGRAM_TOKEN / TELEGRAM_CHAT_ID تنظیم نشده.")

    if state["active"]:
        check_alerts(state, klines_cache)
    else:
        print("هیچ هشدار فعالی وجود ندارد.")

    save_state(state)
    return 0


if __name__ == "__main__":
    sys.exit(main())
