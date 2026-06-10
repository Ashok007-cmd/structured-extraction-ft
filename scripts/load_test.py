#!/usr/bin/env python3
"""Simple async load-test for the structured-extraction API.

Fires a configurable number of concurrent requests at POST /v1/extract and
reports latency percentiles, throughput, and error rate.

Usage:
    python scripts/load_test.py --url http://localhost:8000 --requests 50 --concurrency 5
"""

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path
from typing import List

import httpx

DEFAULT_TEXT = (
    "Acme Corp announced a $50M Series B funding round led by Example "
    "Ventures on March 3, 2026. The new capital will be used to expand "
    "the engineering team and launch the company's flagship product."
)


async def _send_one(client: httpx.AsyncClient, url: str, text: str) -> dict:
    start = time.perf_counter()
    try:
        resp = await client.post(f"{url}/v1/extract", json={"text": text}, timeout=120.0)
        elapsed = time.perf_counter() - start
        return {
            "ok": resp.status_code == 200,
            "status_code": resp.status_code,
            "elapsed_s": elapsed,
        }
    except Exception as e:
        elapsed = time.perf_counter() - start
        return {"ok": False, "status_code": None, "elapsed_s": elapsed, "error": str(e)}


async def run_load_test(url: str, num_requests: int, concurrency: int, text: str) -> dict:
    semaphore = asyncio.Semaphore(concurrency)
    results: List[dict] = []

    async def worker():
        async with semaphore:
            async with httpx.AsyncClient() as client:
                results.append(await _send_one(client, url, text))

    start = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(num_requests)))
    total_time = time.perf_counter() - start

    latencies = sorted(r["elapsed_s"] for r in results)
    successes = [r for r in results if r["ok"]]
    failures = [r for r in results if not r["ok"]]

    def pct(p: float) -> float:
        if not latencies:
            return 0.0
        idx = min(len(latencies) - 1, int(len(latencies) * p))
        return latencies[idx]

    summary = {
        "url": url,
        "num_requests": num_requests,
        "concurrency": concurrency,
        "total_time_s": total_time,
        "throughput_rps": num_requests / total_time if total_time > 0 else 0.0,
        "success_count": len(successes),
        "failure_count": len(failures),
        "latency_s": {
            "min": min(latencies) if latencies else 0.0,
            "mean": statistics.mean(latencies) if latencies else 0.0,
            "p50": pct(0.50),
            "p90": pct(0.90),
            "p99": pct(0.99),
            "max": max(latencies) if latencies else 0.0,
        },
        "failures": failures[:5],
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL of the API")
    parser.add_argument("--requests", type=int, default=20, help="Total number of requests to send")
    parser.add_argument("--concurrency", type=int, default=1, help="Number of concurrent requests")
    parser.add_argument("--text", default=DEFAULT_TEXT, help="Text to send in each request")
    parser.add_argument("--output", default=None, help="Optional path to write JSON report")
    args = parser.parse_args()

    summary = asyncio.run(run_load_test(args.url, args.requests, args.concurrency, args.text))

    print(json.dumps(summary, indent=2))

    if args.output:
        Path(args.output).write_text(json.dumps(summary, indent=2))
        print(f"\nReport written to {args.output}")


if __name__ == "__main__":
    main()
