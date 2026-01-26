#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import json
import os
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import websockets


# ================= Kubernetes / WS Config =================

# Kubernetes Service DNS
WS_URL = os.getenv("WS_URL", "ws://kline-ws:8765")

# Per-symbol rolling window
WINDOW = int(os.getenv("WINDOW", "90"))

# Max WS frame size (must >= server max_size)
MAX_SIZE = int(os.getenv("MAX_SIZE", str(16 * 1024 * 1024)))

RECONNECT_DELAY = 2.0


# ================= Utils =================

def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def pretty_bytes(n: int) -> str:
    for u in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.0f}{u}"
        n /= 1024
    return f"{n:.2f}TB"


# ================= Data Model =================

@dataclass(frozen=True)
class Bar:
    opentime: int
    o: float
    h: float
    l: float
    c: float
    v: float
    e_symbol: str


# ================= Storage =================

class TimeSeriesStore:
    """
    Per symbol:
  OrderedDict[opentime -> Bar], maxlen = WINDOW
    """
    def __init__(self, window: int):
        self.window = window
        self.data: Dict[str, OrderedDict[int, Bar]] = {}

    def upsert(self, bars: List[Bar]) -> Tuple[int, int]:
        changed = 0
        trimmed = 0

        for bar in bars:
            sym = bar.e_symbol
            od = self.data.setdefault(sym, OrderedDict())

            prev = od.get(bar.opentime)
            if prev != bar:
                od[bar.opentime] = bar
                changed += 1

            # ensure order
            if len(od) > 1:
                keys = list(od.keys())
                if keys[-1] != max(keys):
                    od_sorted = OrderedDict(sorted(od.items()))
                    od.clear()
                    od.update(od_sorted)

            while len(od) > self.window:
                od.popitem(last=False)
                trimmed += 1

        return changed, trimmed

    def stats(self) -> dict:
        sym_cnt = len(self.data)
        lens = [len(v) for v in self.data.values()] or [0]
        total = sum(lens)
        approx_bytes = total * 64  # rough

        return {
            "symbols": sym_cnt,
            "total_bars": total,
            "min_window": min(lens),
            "max_window": max(lens),
            "approx_bytes": approx_bytes,
        }


# ================= Payload Parsing =================

def parse_payload(raw: str) -> Tuple[str, List[Bar]]:
    obj = json.loads(raw)
    header = obj.get("header", "")
    records = obj.get("records") or []

    bars: List[Bar] = []
    for r in records:
        if not isinstance(r, list) or len(r) < 7:
            continue
        try:
            bars.append(
                Bar(
                    opentime=int(r[0]),
                    o=float(r[1]),
                    h=float(r[2]),
                    l=float(r[3]),
                    c=float(r[4]),
                    v=float(r[5]),
                    e_symbol=str(r[6]),
                )
            )
        except Exception:
            continue

    return header, bars


# ================= Main Loop =================

async def run():
    print(f"[{utc_now()}] WS collector starting")
    print(f"  WS_URL={WS_URL}")
    print(f"  WINDOW={WINDOW}")

    store = TimeSeriesStore(WINDOW)

    while True:
        try:
            async with websockets.connect(
                WS_URL,
                max_size=MAX_SIZE,
                ping_interval=20,
                ping_timeout=20,
            ) as ws:
                print(f"[{utc_now()}] Connected to {WS_URL}")

                async for raw in ws:
                    payload_bytes = len(raw.encode("utf-8", "ignore"))

                    try:
                        header, bars = parse_payload(raw)
                    except Exception as e:
                        print(f"[{utc_now()}] [WARN] parse error: {e}")
                        continue

                    changed, trimmed = store.upsert(bars)
                    st = store.stats()

                    print(
                        f"[{utc_now()}] [RECV] header={header} "
                        f"payload={pretty_bytes(payload_bytes)} "
                        f"records={len(bars)} upserted={changed} trimmed={trimmed} "
                        f"symbols={st['symbols']} total_bars={st['total_bars']} "
                        f"window[min,max]=[{st['min_window']},{st['max_window']}] "
                        f"approx_store={pretty_bytes(st['approx_bytes'])}"
                    )

        except Exception as e:
            print(f"[{utc_now()}] [WARN] WS disconnected: {e}")
            await asyncio.sleep(RECONNECT_DELAY)


if __name__ == "__main__":
    asyncio.run(run())
