#!/usr/bin/env python3
import argparse
import asyncio
import curses
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

import httpx


@dataclass
class Sample:
    latency_ms: Optional[float]
    status: Optional[int]
    error: Optional[str]


async def probe_once(
    client: httpx.AsyncClient,
    url: str,
    timeout: float,
    method: str,
) -> Sample:
    start = time.perf_counter()
    try:
        response = await client.request(method, url, timeout=timeout)
        await response.aread()
        elapsed_ms = (time.perf_counter() - start) * 1000
        return Sample(latency_ms=elapsed_ms, status=response.status_code, error=None)
    except httpx.HTTPError as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return Sample(latency_ms=elapsed_ms, status=None, error=str(exc))


async def probe(url: str, timeout: float, method: str, concurrency: int) -> Sample:
    async with httpx.AsyncClient(http2=True) as client:
        tasks = [probe_once(client, url, timeout, method) for _ in range(concurrency)]
        results = await asyncio.gather(*tasks)

    ok_samples = [sample for sample in results if sample.status is not None]
    if ok_samples:
        average_latency = sum(sample.latency_ms or 0 for sample in ok_samples) / len(
            ok_samples
        )
        return Sample(
            latency_ms=average_latency,
            status=ok_samples[0].status,
            error=None,
        )

    first_error = next((sample.error for sample in results if sample.error), "unknown")
    return Sample(latency_ms=None, status=None, error=first_error)


def format_latency(latency_ms: Optional[float]) -> str:
    if latency_ms is None:
        return "-"
    if latency_ms < 1000:
        return f"{latency_ms:6.1f} ms"
    return f"{latency_ms / 1000:6.2f} s"


def color_for_status(status: int) -> int:
    if 200 <= status < 300:
        return 1
    if 400 <= status < 500:
        return 2
    if 500 <= status < 600:
        return 3
    return 0


def draw_screen(
    screen: "curses._CursesWindow", # type: ignore
    url: str,
    interval: float,
    concurrency: int,
    samples: Deque[Sample],
) -> None:
    screen.erase()
    rows, cols = screen.getmaxyx()

    title = "ghttpping - HTTP/HTTPS TUI monitor"
    screen.addnstr(0, 0, title, cols - 1)
    screen.addnstr(1, 0, f"Target: {url}", cols - 1)

    ok_latencies = [s.latency_ms for s in samples if s.status is not None]
    last = samples[-1] if samples else None
    min_latency = min(ok_latencies) if ok_latencies else None # type: ignore
    max_latency = max(ok_latencies) if ok_latencies else None # type: ignore

    status_text = ""
    if last is not None:
        if last.status is not None:
            status_text = f"HTTP {last.status}"
        else:
            status_text = "ERR"
    summary = (
        f"Interval: {interval:.1f}s | Concurrency: {concurrency} | "
        f"Last: {format_latency(last.latency_ms) if last else '-'}"
        f" | Min: {format_latency(min_latency)} | Max: {format_latency(max_latency)} | {status_text}"
    )
    screen.addnstr(2, 0, summary, cols - 1)

    graph_top = 4
    graph_height = max(rows - graph_top - 1, 1)
    graph_width = max(cols - 2, 1)

    max_scale = max_latency or 1.0
    if max_scale <= 0:
        max_scale = 1.0

    start_index = max(len(samples) - graph_width, 0)
    graph_samples = list(samples)[start_index:]

    for x, sample in enumerate(graph_samples):
        if sample.status is None:
            y = rows - 2
            if y >= graph_top:
                screen.addch(y, x + 1, "x")
            continue
        height = int((sample.latency_ms or 0) / max_scale * (graph_height - 1))
        height = max(height, 0)
        bar_attr = curses.color_pair(color_for_status(sample.status))
        for offset in range(height + 1):
            y = rows - 2 - offset
            if y < graph_top:
                break
            screen.addch(y, x + 1, "█", bar_attr)

    scale_label = f"max {format_latency(max_latency)}"
    screen.addnstr(graph_top - 1, 0, scale_label, cols - 1)
    screen.addnstr(rows - 1, 0, "Press Ctrl+C to exit", cols - 1)
    screen.refresh()


def run_monitor(url: str, interval: float, timeout: float, method: str, concurrency: int) -> None:
    samples: Deque[Sample] = deque(maxlen=500)

    def _loop(screen: "curses._CursesWindow") -> None: # type: ignore
        async def _async_loop() -> None:
            curses.curs_set(0)
            if curses.has_colors():
                curses.start_color()
                curses.use_default_colors()
                curses.init_pair(1, curses.COLOR_GREEN, -1)
                curses.init_pair(2, curses.COLOR_YELLOW, -1)
                curses.init_pair(3, curses.COLOR_RED, -1)

            screen.nodelay(True)
            while True:
                sample = await probe(url, timeout, method, concurrency)
                samples.append(sample)
                draw_screen(screen, url, interval, concurrency, samples)
                await asyncio.sleep(interval)

        asyncio.run(_async_loop())

    curses.wrapper(_loop)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HTTP/HTTPS echo/reply monitor with TUI graph output."
    )
    parser.add_argument("url", help="Target URL (http/https)")
    parser.add_argument(
        "-i",
        "--interval",
        type=float,
        default=1.0,
        help="Interval between requests in seconds (default: 1.0)",
    )
    parser.add_argument(
        "-t",
        "--timeout",
        type=float,
        default=1.0,
        help="Request timeout in seconds (default: 1.0)",
    )
    parser.add_argument(
        "-m",
        "--method",
        choices=["GET", "HEAD"],
        default="GET",
        help="HTTP method to use (default: GET)",
    )
    parser.add_argument(
        "-c",
        "--concurrency",
        type=int,
        default=1,
        help="Concurrent requests per interval (default: 1)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_monitor(args.url, args.interval, args.timeout, args.method, args.concurrency)


if __name__ == "__main__":
    main()
