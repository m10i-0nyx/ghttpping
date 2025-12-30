#!/usr/bin/env python3
import argparse
import asyncio
import curses
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Dict
import socket
import ipaddress
from urllib.parse import urlsplit
from datetime import datetime

import httpx


@dataclass
class Sample:
    latency_ms: Optional[float]
    status: Optional[int]
    error: Optional[str]
    peer_ip: Optional[str] = None
    http_version: Optional[str] = None


async def probe_once(
    client: httpx.AsyncClient,
    url: str,
    timeout: float,
    method: str,
    headers: Optional[Dict[str, str]] = None,
) -> Sample:
    start = time.perf_counter()
    try:
        response = await client.request(method, url, headers=headers, timeout=timeout)
        await response.aread()
        elapsed_ms = (time.perf_counter() - start) * 1000
        http_ver = getattr(response, "http_version", None)
        return Sample(latency_ms=elapsed_ms, status=response.status_code, error=None, peer_ip=None, http_version=http_ver)
    except httpx.HTTPError as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return Sample(latency_ms=elapsed_ms, status=None, error=str(exc), peer_ip=None, http_version=None)


async def fetch_public_ip(api_url: str, timeout: float = 1.0) -> Optional[str]:
    try:
        async with httpx.AsyncClient(http2=True) as client:
            r = await client.get(api_url, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            return data.get("client_host")
    except Exception:
        return None


async def probe(
    url: str,
    timeout: float,
    method: str,
    concurrency: int,
    ip_version: Optional[int] = None,
    user_agent: Optional[str] = None,
    insecure: bool = False,
) -> Sample:
    effective_url = url  # keep original host in URL to preserve SNI
    headers: Optional[Dict[str, str]] = None
    resolved_ip: Optional[str] = None

    parsed = urlsplit(url)
    host = parsed.hostname
    if host is None:
        return Sample(latency_ms=None, status=None, error="invalid URL", peer_ip=None, http_version=None)

    loop = asyncio.get_running_loop()

    # Resolve an address for display / checking, but DO NOT replace URL (to keep SNI)
    if ip_version is not None:
        try:
            addr_obj = ipaddress.ip_address(host)
            # host is a literal IP
            if (addr_obj.version == 4 and ip_version == 6) or (
                addr_obj.version == 6 and ip_version == 4
            ):
                return Sample(
                    latency_ms=None,
                    status=None,
                    error=f"host is {addr_obj.version}, but -{ip_version} was requested",
                    peer_ip=None,
                    http_version=None,
                )
            resolved_ip = host
        except ValueError:
            family = socket.AF_INET if ip_version == 4 else socket.AF_INET6
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
            try:
                infos = await loop.getaddrinfo(host, str(port), family=family, type=socket.SOCK_STREAM)
                if not infos:
                    return Sample(latency_ms=None, status=None, error="no address found", peer_ip=None, http_version=None)
                resolved_ip = infos[0][4][0]
            except Exception as exc:
                return Sample(latency_ms=None, status=None, error=str(exc), peer_ip=None, http_version=None)
    else:
        # best-effort resolution for display only
        try:
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
            infos = await loop.getaddrinfo(host, str(port), family=socket.AF_UNSPEC, type=socket.SOCK_STREAM)
            if infos:
                resolved_ip = infos[0][4][0]
        except Exception:
            resolved_ip = None

    # Prepare headers (do not override Host header; URL host remains original)
    headers = {}
    if user_agent:
        headers["User-Agent"] = user_agent

    # If ip_version is set, monkeypatch socket.getaddrinfo temporarily to force address family.
    orig_getaddrinfo = None
    if ip_version is not None:
        orig_getaddrinfo = socket.getaddrinfo

        desired_family = socket.AF_INET if ip_version == 4 else socket.AF_INET6

        def patched_getaddrinfo(hostname, port, family=0, type=0, proto=0, flags=0):
            results = orig_getaddrinfo(hostname, port, family, type, proto, flags)
            if family == socket.AF_UNSPEC or family == 0:
                filtered = [r for r in results if r[0] == desired_family]
                return filtered
            return results

        socket.getaddrinfo = patched_getaddrinfo  # type: ignore

    try:
        async with httpx.AsyncClient(http2=True, verify=not insecure) as client:
            tasks = [
                probe_once(client, effective_url, timeout, method, headers=headers)
                for _ in range(concurrency)
            ]
            results = await asyncio.gather(*tasks)
    finally:
        if orig_getaddrinfo is not None:
            socket.getaddrinfo = orig_getaddrinfo  # restore

    ok_samples = [sample for sample in results if sample.status is not None]
    if ok_samples:
        average_latency = sum(sample.latency_ms or 0 for sample in ok_samples) / len(
            ok_samples
        )
        # pick protocol from first successful response
        http_ver = ok_samples[0].http_version
        return Sample(
            latency_ms=average_latency,
            status=ok_samples[0].status,
            error=None,
            peer_ip=resolved_ip,
            http_version=http_ver,
        )

    first_error = next((sample.error for sample in results if sample.error), "unknown")
    return Sample(latency_ms=None, status=None, error=first_error, peer_ip=resolved_ip, http_version=None)


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
    screen: "curses._CursesWindow",  # type: ignore
    url: str,
    interval: float,
    concurrency: int,
    samples: Deque[Sample],
    public_ips: Dict[str, Optional[str]],
    last_conn_time: Optional[str],
) -> None:
    screen.erase()
    rows, cols = screen.getmaxyx()

    title = "ghttpping - HTTP/HTTPS TUI monitor"
    screen.addnstr(0, 0, title, cols - 1)

    last = samples[-1] if samples else None
    peer_ip_text = last.peer_ip if last and last.peer_ip else "-"
    # show target + resolved peer IP
    screen.addnstr(1, 0, f"Target: {url} ({peer_ip_text})", cols - 1)

    # show user's global IPs (IPv4/IPv6)
    v4 = public_ips.get("v4") or "-"
    v6 = public_ips.get("v6") or "-"
    screen.addnstr(2, 0, f"Your Global IPs: v4={v4}  v6={v6}", cols - 1)

    ok_latencies = [s.latency_ms for s in samples if s.status is not None]
    min_latency = min(ok_latencies) if ok_latencies else None  # type: ignore
    max_latency = max(ok_latencies) if ok_latencies else None  # type: ignore

    status_text = ""
    protocol_text = ""
    error_text = "-"
    if last is not None:
        if last.status is not None:
            status_text = f"HTTP {last.status}"
        else:
            status_text = "ERR"
        protocol_text = last.http_version or "-"
        if last.error:
            error_text = last.error

    summary = (
        f"Interval: {interval:.1f}s | Concurrency: {concurrency} | "
        f"Proto: {protocol_text} | Last: {format_latency(last.latency_ms) if last else '-'}"
        f" | Min: {format_latency(min_latency)} | Max: {format_latency(max_latency)} | {status_text}"
    )
    screen.addnstr(3, 0, summary, cols - 1)

    # show error on its own line
    screen.addnstr(4, 0, f"Err: {error_text}", cols - 1)

    graph_top = 5
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
    screen.addnstr(graph_top, 0, scale_label, cols - 1)
    # left message
    screen.addnstr(rows - 1, 0, "Press Ctrl+C to exit", cols - 1)
    # right: last connection time
    if last_conn_time:
        rtxt = last_conn_time
    else:
        rtxt = "-"
    col_pos = max(cols - len(rtxt) - 1, 0)
    screen.addnstr(rows - 1, col_pos, rtxt, cols - col_pos - 1)
    screen.refresh()


def run_monitor(url: str, interval: float, timeout: float, method: str, concurrency: int, ip_version: Optional[int], user_agent: Optional[str]) -> None:
    samples: Deque[Sample] = deque(maxlen=500)

    def _loop(screen: "curses._CursesWindow") -> None:  # type: ignore
        async def _async_loop() -> None:
            curses.curs_set(0)
            if curses.has_colors():
                curses.start_color()
                curses.use_default_colors()
                curses.init_pair(1, curses.COLOR_GREEN, -1)
                curses.init_pair(2, curses.COLOR_YELLOW, -1)
                curses.init_pair(3, curses.COLOR_RED, -1)

            screen.nodelay(True)

            public_ips: Dict[str, Optional[str]] = {"v4": None, "v6": None}
            last_conn_time: Optional[str] = None

            async def refresh_public_ips_loop() -> None:
                # initial immediate fetch
                while True:
                    try:
                        public_ips["v4"] = await fetch_public_ip("https://getipv4.0nyx.net/json")
                        public_ips["v6"] = await fetch_public_ip("https://getipv6.0nyx.net/json")
                        await asyncio.sleep(10)
                    except Exception:
                        await asyncio.sleep(10)

            asyncio.create_task(refresh_public_ips_loop())

            while True:
                sample = await probe(url, timeout, method, concurrency, ip_version=ip_version, user_agent=user_agent)
                samples.append(sample)
                # update last connection time when probe completed
                last_conn_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                draw_screen(screen, url, interval, concurrency, samples, public_ips, last_conn_time)
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
    parser.add_argument(
        "-U",
        "--user-agent",
        dest="user_agent",
        default=None,
        help="Set User-Agent header (default: httpx default)",
    )
    parser.add_argument(
        "-k",
        "--insecure",
        action="store_true",
        help="Skip SSL certificate verification",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-4",
        dest="ip_version",
        action="store_const",
        const=4,
        help="Force IPv4 name resolution",
    )
    group.add_argument(
        "-6",
        dest="ip_version",
        action="store_const",
        const=6,
        help="Force IPv6 name resolution",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_monitor(args.url, args.interval, args.timeout, args.method, args.concurrency, args.ip_version, args.user_agent)


if __name__ == "__main__":
    main()
