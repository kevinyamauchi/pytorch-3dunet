#!/usr/bin/env python3
"""Append RAM + GPU samples until root_pid exits (or timeout)."""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path


def _children_by_parent() -> dict[int, list[int]]:
    children: dict[int, list[int]] = {}
    proc_dir = Path("/proc")
    for entry in proc_dir.iterdir():
        if not entry.name.isdigit():
            continue
        status_path = entry / "status"
        try:
            ppid = None
            for line in status_path.read_text().splitlines():
                if line.startswith("PPid:"):
                    ppid = int(line.split()[1])
                    break
            if ppid is not None:
                children.setdefault(ppid, []).append(int(entry.name))
        except (OSError, ValueError):
            continue
    return children


def _process_tree(root_pid: int) -> list[int]:
    children = _children_by_parent()
    stack = [root_pid]
    out: list[int] = []
    while stack:
        pid = stack.pop()
        out.append(pid)
        stack.extend(children.get(pid, []))
    return out


def _rss_kb(pid: int) -> int:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except OSError:
        return 0
    return 0


def _gpu_query() -> tuple[str, str]:
    try:
        r = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        line = (r.stdout or "").strip().splitlines()[0]
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            return parts[0], f"{parts[1]}/{parts[2]}"
    except (OSError, IndexError, subprocess.TimeoutExpired):
        pass
    return "", ""


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("root_pid", type=int)
    p.add_argument("out_csv", type=Path)
    p.add_argument("--interval", type=float, default=10.0)
    p.add_argument("--timeout", type=float, default=0.0, help="0 = no timeout")
    args = p.parse_args()

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    new_file = not args.out_csv.exists()
    with args.out_csv.open("a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(
                [
                    "elapsed_s",
                    "pid_count",
                    "total_rss_mb",
                    "gpu_util_percent",
                    "gpu_memory_mib",
                ]
            )
        t0 = time.monotonic()
        while True:
            if not Path(f"/proc/{args.root_pid}").exists():
                break
            tree = _process_tree(args.root_pid)
            rss = sum(_rss_kb(pid) for pid in tree) / 1024
            util, mem = _gpu_query()
            elapsed = time.monotonic() - t0
            w.writerow([f"{elapsed:.3f}", len(tree), f"{rss:.3f}", util, mem])
            f.flush()
            if args.timeout > 0 and elapsed >= args.timeout:
                break
            time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    sys.exit(main())
