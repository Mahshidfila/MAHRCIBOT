"""
هشدار چندتایم‌فریمی RCI برای BTCUSDT
--------------------------------------
منطق:
۱) رژیم فعلی بازار از روی آخرین عبور RCI(81) در تایم‌فریم ۱ ساعته تعیین می‌شه:
   - اگه آخرین عبور، صعودی از +۶۰ بوده  -> رژیم "صعودی"
   - اگه آخرین عبور، نزولی از -۶۰ بوده  -> رژیم "نزولی"

۲) بسته به رژیم، تایم‌فریم ۵ دقیقه‌ی RCI(81) چک می‌شه:
   - رژیم نزولی: هر بار RCI81(5m) به صورت نزولی به ۸۰ برسه   -> هشدار
   - رژیم صعودی: هر بار RCI81(5m) به صورت صعودی به -۸۰ برسه  -> هشدار

۳) هر بار هشدار ۵ دقیقه صادر شد، مقدار RCI(81) در تایم‌فریم ۱۵ دقیقه
   (نه ۵ دقیقه) هم چک و در پیام اعلام می‌شه:
   - رژیم نزولی: آیا RCI81(15m) > +۶۰ هست یا نه
   - رژیم صعودی: آیا RCI81(15m) < -۶۰ هست یا نه

نکته‌ی مهم: این اسکریپت به هیچ ذخیره‌سازی وضعیتی بین اجراها نیاز نداره.
هر بار که اجرا می‌شه، رژیم رو از صفر و از روی تاریخچه‌ی تازه‌ی ۱ ساعته
پیدا می‌کنه، و کراس ۵ دقیقه رو هم فقط بین دو کندل آخر (تازه‌بسته‌شده)
چک می‌کنه. بنابراین باید هر ۵ دقیقه اجرا بشه تا هیچ کراسی از قلم نیفته.
"""

import os
import sys
import numpy as np
import pandas as pd
import requests

SYMBOL = "BTCUSDT"
RCI_LENGTH = 81  # همون طول ۸۱ روی هر سه تایم‌فریم استفاده می‌شه

H1_INTERVAL = "1h"
H1_UPPER = 60
H1_LOWER = -60
H1_LOOKBACK = 500  # حدود ۲۰ روز، برای پیدا کردن آخرین عبور رژیم

M5_INTERVAL = "5m"
M5_UPPER = 80
M5_LOWER = -80
M5_LOOKBACK = 200

M15_INTERVAL = "15m"
M15_UPPER = 60
M15_LOWER = -60
M15_LOOKBACK = 150

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

BASE_URL = "https://data-api.binance.vision/api/v3/klines"


def fetch_klines(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    resp = requests.get(BASE_URL, params=params, timeout=10)
    if resp.status_code != 200:
        print(f"خطا در واکشی {interval} - کد وضعیت: {resp.status_code}")
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
    # آخرین کندل ممکنه هنوز بسته نشده باشه؛ حذفش می‌کنیم
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
        sum_d2 = np.sum(d ** 2)
        values[end] = (1 - (6 * sum_d2) / (length * (length ** 2 - 1))) * 100
    return pd.Series(values, index=series.index)


def find_last_regime(rci: pd.Series, upper: float, lower: float):
    """آخرین رژیم رو از روی آخرین عبور صعودی از upper یا نزولی از lower پیدا می‌کنه."""
    last_up = -1
    last_down = -1
    for i in range(1, len(rci)):
        prev, curr = rci.iloc[i - 1], rci.iloc[i]
        if pd.isna(prev) or pd.isna(curr):
            continue
        if prev <= upper < curr:
            last_up = i
        if prev >= lower > curr:
            last_down = i
    if last_up == -1 and last_down == -1:
        return None
    return "bullish" if last_up > last_down else "bearish"


def send_telegram(message: str) -> None:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️  TELEGRAM_TOKEN یا TELEGRAM_CHAT_ID تنظیم نشده؛ پیام ارسال نشد.")
        print(message)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    resp = requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": message}, timeout=10)
    resp.raise_for_status()


def main() -> int:
    # --- مرحله ۱: تعیین رژیم از روی تایم‌فریم ۱ ساعته ---
    df_1h = fetch_klines(SYMBOL, H1_INTERVAL, H1_LOOKBACK)
    df_1h["rci"] = calc_rci(df_1h["close"], RCI_LENGTH)
    regime = find_last_regime(df_1h["rci"], H1_UPPER, H1_LOWER)

    if regime is None:
        print("در بازه‌ی داده‌ی موجود، عبور معتبری از ±۶۰ روی تایم‌فریم ۱ ساعته پیدا نشد.")
        return 0

    print(f"رژیم فعلی (بر اساس آخرین عبور RCI81 در ۱ ساعته): {regime}")

    # --- مرحله ۲: چک تریگر تایم‌فریم ۵ دقیقه ---
    df_5m = fetch_klines(SYMBOL, M5_INTERVAL, M5_LOOKBACK)
    df_5m["rci"] = calc_rci(df_5m["close"], RCI_LENGTH)

    if df_5m["rci"].iloc[-2:].isna().any():
        print("داده‌ی کافی برای محاسبه‌ی RCI در تایم‌فریم ۵ دقیقه وجود ندارد.")
        return 1

    prev_5m = df_5m["rci"].iloc[-2]
    curr_5m = df_5m["rci"].iloc[-1]
    candle_time_5m = df_5m["open_time"].iloc[-1]

    triggered = False
    if regime == "bearish":
        triggered = prev_5m >= M5_UPPER > curr_5m
    elif regime == "bullish":
        triggered = prev_5m <= M5_LOWER < curr_5m

    if not triggered:
        print(f"بدون سیگنال ۵ دقیقه‌ای. RCI81(5m): {curr_5m:.2f}")
        return 0

    # --- مرحله ۳: فیلتر تأییدی روی تایم‌فریم ۱۵ دقیقه ---
    df_15m = fetch_klines(SYMBOL, M15_INTERVAL, M15_LOOKBACK)
    df_15m["rci"] = calc_rci(df_15m["close"], RCI_LENGTH)

    if pd.isna(df_15m["rci"].iloc[-1]):
        print("داده‌ی کافی برای محاسبه‌ی RCI در تایم‌فریم ۱۵ دقیقه وجود ندارد.")
        return 1

    curr_15m = df_15m["rci"].iloc[-1]

    if regime == "bearish":
        filter_ok = curr_15m > M15_UPPER
        message = (
            f"🟠 {SYMBOL} — سیگنال ۵ دقیقه‌ای (رژیم نزولی، بر اساس 1H به -۶۰ رسیده)\n"
            f"RCI81(5m) به صورت نزولی به {M5_UPPER} رسید: {curr_5m:.2f}\n"
            f"RCI81(15m) فعلی: {curr_15m:.2f} → "
            f"{'بله، بیشتر از +۶۰ است' if filter_ok else 'خیر، بیشتر از +۶۰ نیست'}\n"
            f"زمان کندل ۵ دقیقه: {candle_time_5m}"
        )
    else:  # bullish
        filter_ok = curr_15m < M15_LOWER
        message = (
            f"🟣 {SYMBOL} — سیگنال ۵ دقیقه‌ای (رژیم صعودی، بر اساس 1H به +۶۰ رسیده)\n"
            f"RCI81(5m) به صورت صعودی به {M5_LOWER} رسید: {curr_5m:.2f}\n"
            f"RCI81(15m) فعلی: {curr_15m:.2f} → "
            f"{'بله، کمتر از -۶۰ است' if filter_ok else 'خیر، کمتر از -۶۰ نیست'}\n"
            f"زمان کندل ۵ دقیقه: {candle_time_5m}"
        )

    send_telegram(message)
    return 0


if __name__ == "__main__":
    sys.exit(main())
