import time
import os
import requests
from zipfile import ZipFile
from io import BytesIO
from collections import defaultdict
from threading import Lock
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# =========================
# 全局统计
# =========================
symbol_download_counts = defaultdict(int)
download_lock = Lock()

# =========================
# 配置
# =========================
FREQUENCY = "daily"  # daily | monthly
INTERVAL = "1m"

TARGET_YEAR = "2026"

BASE_DATA_URL = "https://data.binance.vision/data/futures/um"
EXCHANGE_INFO_URL = "https://fapi.binance.com/fapi/v1/exchangeInfo"

DOWNLOAD_DIR = f"/data/raw_data/binance_1m_{FREQUENCY}_klines_futures"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

CURRENT_YEAR = datetime.now().year
CURRENT_MONTH = datetime.now().month
CURRENT_DAY = datetime.now().day


# =========================
# 1️⃣ 获取 USDT 永续合约 symbols（官方 API）
# =========================
def get_usdt_perp_symbols():
    print("📡 Fetching symbols from Binance Futures API...")
    r = requests.get(EXCHANGE_INFO_URL, timeout=10)
    r.raise_for_status()

    data = r.json()

    symbols = [
        s["symbol"]
        for s in data["symbols"]
        if s["quoteAsset"] == "USDT"
        and s["contractType"] == "PERPETUAL"
        and s["status"] == "TRADING"
    ]

    print(f"✅ Found {len(symbols)} USDT perpetual symbols")
    return symbols


# =========================
# 2️⃣ 构造 zip 下载 URL
# =========================
def build_urls(symbols, year=TARGET_YEAR, interval=INTERVAL):
    urls = []

    for symbol in symbols:
        base = f"{BASE_DATA_URL}/{FREQUENCY}/klines/{symbol}/{interval}/"

        for month in range(1, 13):
            for day in range(1, 32):
                if (month == CURRENT_MONTH and day >= CURRENT_DAY) or month>CURRENT_MONTH:
                    continue

                urls.append(
                    f"{base}{symbol}-{interval}-{year}-{month:02d}-{day:02d}.zip"
                )

    return urls


# =========================
# 3️⃣ 下载 & 解压
# =========================
def download_zip_from_url(zip_url):
    filename = zip_url.split("/")[-1]
    symbol = zip_url.split("/")[-3]

    symbol_dir = os.path.join(DOWNLOAD_DIR, symbol)
    os.makedirs(symbol_dir, exist_ok=True)

    zip_path = os.path.join(symbol_dir, filename.replace(".zip", ""))

    # =========================
    # ✅ 已存在就跳过
    # =========================
    if os.path.exists(zip_path):
        print(f"⏭️  Skip existing {symbol}/{filename}")
        return

    print(f"⬇️  {symbol} | {filename}")

    try:
        r = requests.get(zip_url, timeout=20)
        if r.status_code == 200:
            with ZipFile(BytesIO(r.content)) as zip_file:
                zip_file.extractall(symbol_dir)

            with download_lock:
                symbol_download_counts[symbol] += 1

            print(f"  ✅ Extracted {filename}")
        else:
            print(f"  ❌ HTTP {r.status_code} - {filename}")

    except Exception as e:
        print(f"  ⚠️ Error {filename}: {e}")



# =========================
# 4️⃣ 主流程
# =========================
def main():
    symbols = get_usdt_perp_symbols()

    # 如需手动限制（调试用）
    # symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

    urls = build_urls(symbols)
    total = len(urls)

    print(f"🚀 Start downloading {total} files...")
    start_time = time.time()
    completed = 0

    with ThreadPoolExecutor(max_workers=10) as executor:
        future_map = {
            executor.submit(download_zip_from_url, url): url
            for url in urls
        }

        for future in as_completed(future_map):
            completed += 1
            elapsed = time.time() - start_time
            avg = elapsed / completed
            eta = avg * (total - completed)
            print(f"📈 Progress {completed}/{total} | ETA {eta:.1f}s")

    print("\n📊 Download summary:")
    for symbol in sorted(symbol_download_counts):
        print(f"🔹 {symbol}: {symbol_download_counts[symbol]} files")


if __name__ == "__main__":
    main()
