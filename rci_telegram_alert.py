"""
هشدار RCI برای BTCUSDT - تایم‌فریم ۱ ساعته
------------------------------------------
- دیتا: API عمومی و رایگان بایننس (بدون نیاز به کلید API)
- شرط هشدار: عبور RCI(81) از بالای +60 یا پایین -60
- ارسال پیام: از طریق Telegram Bot API

نحوه‌ی اجرا:
    python rci_telegram_alert.py

متغیرهای محیطی لازم:
    TELEGRAM_TOKEN    - توکن ربات تلگرام (از BotFather)
    TELEGRAM_CHAT_ID  - آیدی چت/کانال مقصد
"""

import os
import sys
import numpy as np
import pandas as pd
import requests

# ------------------ تنظیمات قابل تغییر ------------------
SYMBOL = "BTCUSDT"
INTERVAL = "1h"
RCI_LENGTH = 81
UPPER_LEVEL = 60
LOWER_LEVEL = -60
LOOKBACK = 200  # تعداد کندل واکشی‌شده (باید به‌وضوح بیشتر از RCI_LENGTH باشد)
# ----------------------------------------------------------

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")


def fetch_klines(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    """واکشی کندل‌ها از دامنه‌ی عمومی و بدون محدودیت جغرافیایی بایننس (بدون نیاز به کلید)."""
    url = "https://data-api.binance.vision/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    resp = requests.get(url, params=params, timeout=10)
    if resp.status_code != 200:
        print(f"خطا در واکشی دیتا - کد وضعیت: {resp.status_code}")
        print(f"پاسخ سرور: {resp.text[:500]}")
    resp.raise_for_status()
    data = resp.json()

    df = pd.DataFrame(
        data,
        columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "qav", "trades", "tb_base", "tb_quote", "ignore",
        ],
    )
    df["close"] = df["close"].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    return df


def calc_rci(series: pd.Series, length: int) -> pd.Series:
    """محاسبه‌ی RCI (Rank Correlation Index) روی یک سری قیمت بسته‌شدن."""
    n = len(series)
    values = np.full(n, np.nan)
    prices = series.values

    for end in range(length - 1, n):
        window = prices[end - length + 1: end + 1]
        date_rank = np.arange(length)
        price_rank = pd.Series(window).rank(method="first").values - 1
        d = date_rank - price_rank
        sum_d2 = np.sum(d ** 2)
        values[end] = (1 - (6 * sum_d2) / (length * (length ** 2 - 1))) * 100

    return pd.Series(values, index=series.index)


def send_telegram(message: str) -> None:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️  TELEGRAM_TOKEN یا TELEGRAM_CHAT_ID تنظیم نشده؛ پیام ارسال نشد.")
        print(message)
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    resp = requests.post(url, data=payload, timeout=10)
    resp.raise_for_status()


def main() -> int:
    df = fetch_klines(SYMBOL, INTERVAL, LOOKBACK)

    # آخرین کندل ممکن است هنوز بسته نشده باشد؛ حذفش می‌کنیم تا فقط کندل‌های
    # کامل و بسته‌شده در محاسبه لحاظ شوند.
    df = df.iloc[:-1].reset_index(drop=True)

    df["rci"] = calc_rci(df["close"], RCI_LENGTH)

    if df["rci"].iloc[-2:].isna().any():
        print("داده‌ی کافی برای محاسبه‌ی RCI با این طول وجود ندارد.")
        return 1

    prev_rci = df["rci"].iloc[-2]
    curr_rci = df["rci"].iloc[-1]
    last_close_time = df["open_time"].iloc[-1]

    cross_up = prev_rci <= UPPER_LEVEL < curr_rci
    cross_down = prev_rci >= LOWER_LEVEL > curr_rci

    if cross_up:
        send_telegram(
            f"🔴 {SYMBOL} - RCI({RCI_LENGTH}, {INTERVAL})\n"
            f"از پایین به بالای {UPPER_LEVEL} عبور کرد\n"
            f"RCI فعلی: {curr_rci:.2f}\n"
            f"زمان کندل: {last_close_time}"
        )
    elif cross_down:
        send_telegram(
            f"🟢 {SYMBOL} - RCI({RCI_LENGTH}, {INTERVAL})\n"
            f"از بالا به پایین {LOWER_LEVEL} عبور کرد\n"
            f"RCI فعلی: {curr_rci:.2f}\n"
            f"زمان کندل: {last_close_time}"
        )
    else:
        print(f"بدون سیگنال. RCI فعلی: {curr_rci:.2f} (زمان کندل: {last_close_time})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
