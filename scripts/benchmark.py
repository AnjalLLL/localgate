"""Benchmark a running localgate instance.

    uv run python scripts/benchmark.py --requests 50 --concurrency 5

Measures the *gateway's* overhead, not the model's speed. To separate the two, run it once
against localgate and once against the backend directly (`--direct http://localhost:11434`)
and compare: the difference is what localgate costs you.

The honest expectation: with memory enabled, the gateway adds one embedding call for
retrieval on the way in and one per chunk on the way out. Against a local embedding model
that is single-digit milliseconds — small next to inference, but not zero, and worth being
able to see.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from dataclasses import dataclass

import httpx


@dataclass
class Result:
    latency_ms: float
    ok: bool
    status: int
    tokens: int = 0


async def one_request(
    client: httpx.AsyncClient, url: str, headers: dict[str, str], body: dict
) -> Result:
    started = time.perf_counter()
    try:
        resp = await client.post(url, json=body, headers=headers, timeout=300.0)
    except httpx.HTTPError:
        # A connection failure still took time; report it rather than dropping the sample.
        return Result((time.perf_counter() - started) * 1000, ok=False, status=0)

    elapsed = (time.perf_counter() - started) * 1000
    tokens = 0
    if resp.status_code == 200:
        tokens = (resp.json().get("usage") or {}).get("total_tokens", 0)
    return Result(elapsed, ok=resp.status_code == 200, status=resp.status_code, tokens=tokens)


async def run(args: argparse.Namespace) -> None:
    url = f"{args.direct or args.url}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if not args.direct:
        headers["Authorization"] = f"Bearer {args.key}"
    if args.session:
        headers["X-Session-ID"] = args.session

    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "stream": False,
    }

    semaphore = asyncio.Semaphore(args.concurrency)

    async def guarded(client: httpx.AsyncClient) -> Result:
        async with semaphore:
            return await one_request(client, url, headers, body)

    print(f"Target      {url}")
    print(f"Requests    {args.requests} at concurrency {args.concurrency}")
    print(f"Model       {args.model}\n")

    started = time.perf_counter()
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(*(guarded(client) for _ in range(args.requests)))
    wall = time.perf_counter() - started

    ok = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]

    if not ok:
        codes = {r.status for r in failed}
        print(f"All {len(failed)} requests failed. Status codes: {codes or 'connection error'}")
        return

    latencies = sorted(r.latency_ms for r in ok)
    tokens = sum(r.tokens for r in ok)

    def pct(p: float) -> float:
        return latencies[min(int(len(latencies) * p), len(latencies) - 1)]

    failures = f"  ({len(failed)} failed)" if failed else ""
    print(f"  succeeded   {len(ok)}/{args.requests}{failures}")
    print(f"  wall time   {wall:.2f}s")
    print(f"  throughput  {len(ok) / wall:.2f} req/s")
    if tokens:
        print(f"  tokens/s    {tokens / wall:.1f}")
    print()
    print(f"  mean        {statistics.mean(latencies):8.1f} ms")
    print(f"  median      {statistics.median(latencies):8.1f} ms")
    print(f"  p95         {pct(0.95):8.1f} ms")
    print(f"  p99         {pct(0.99):8.1f} ms")
    print(f"  min / max   {latencies[0]:.1f} / {latencies[-1]:.1f} ms")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000", help="localgate base URL")
    parser.add_argument(
        "--direct",
        help="Benchmark the backend directly (e.g. http://localhost:11434), skipping localgate. "
        "Compare with a normal run to see the gateway's overhead.",
    )
    parser.add_argument("--key", default="lg_your_key_here", help="A localgate API key")
    parser.add_argument("--model", default="llama3")
    parser.add_argument("--prompt", default="Say hello in exactly five words.")
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--session",
        help="Send an X-Session-ID, exercising the memory path. Omit to benchmark without it.",
    )
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
