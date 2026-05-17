#!/usr/bin/env python3
"""Generate and run StimCycle training memory benchmarks.

The script intentionally has no psutil dependency so it can run in the same
environment as the training job. It samples the full process tree RSS via /proc.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Iterable

import yaml


DEFAULT_DEBUG_DIR = Path("/local1/lfranz/local_data/debug_training_memory")
DEFAULT_SOURCE_CONFIG = DEFAULT_DEBUG_DIR / "follicle_3dunet_training_config.yaml"
DEFAULT_ZARR_ROOT = DEFAULT_DEBUG_DIR / "combined_zarr"
DEFAULT_RUNS_DIR = DEFAULT_DEBUG_DIR / "runs"
CLUSTER_ZARR_ROOT = "/cluster/project/cobi/leopold/ivf/nobackup/validated_raw/StimCycle/combined_zarr"


def _load_yaml(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.load(f, Loader=yaml.SafeLoader)


def _write_yaml(path: Path, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def _local_zarr_paths(zarr_root: Path) -> list[str]:
    paths = [
        str(path)
        for path in sorted(zarr_root.iterdir())
        if path.is_dir() and path.name.endswith(".zarr")
    ]
    if not paths:
        raise RuntimeError(f"No .zarr stores found under {zarr_root}")
    return paths


def _localize_paths(paths: Iterable[str], zarr_root: Path) -> list[str]:
    local_paths = []
    for path in paths:
        basename = Path(path).name
        local_path = zarr_root / basename
        if local_path.is_dir():
            local_paths.append(str(local_path))
    return local_paths


def _choose_paths(source_paths: list[str], all_local_paths: list[str], count: int) -> list[str]:
    existing = _localize_paths(source_paths, Path(all_local_paths[0]).parent)
    candidates = existing or all_local_paths
    if len(candidates) < count:
        raise RuntimeError(
            f"Requested {count} files, but only {len(candidates)} matching local zarr stores are available"
        )
    return candidates[:count]


def _choose_all_paths(source_paths: list[str], zarr_root: Path, all_local_paths: list[str]) -> list[str]:
    existing = _localize_paths(source_paths, zarr_root)
    return existing or all_local_paths


def make_config(
    source_config: Path,
    zarr_root: Path,
    output_dir: Path,
    file_count: int | None,
    batch_size: int,
    num_workers: int,
    max_iterations: int | None,
    run_name: str,
    raw_dtype: str | None = None,
    prefetch_factor: int | None = None,
    multiprocessing_context: str | None = None,
    full_training: bool = False,
) -> Path:
    config = _load_yaml(source_config)
    all_local_paths = _local_zarr_paths(zarr_root)
    train_source_paths = config["loaders"]["train"]["file_paths"]
    val_source_paths = config["loaders"]["val"]["file_paths"]

    if full_training:
        train_paths = _choose_all_paths(train_source_paths, zarr_root, all_local_paths)
    else:
        assert file_count is not None
        train_paths = _choose_paths(train_source_paths, all_local_paths, file_count)
    train_set = set(train_paths)

    val_paths = [path for path in _localize_paths(val_source_paths, zarr_root) if path not in train_set]
    if not val_paths:
        val_paths = [path for path in all_local_paths if path not in train_set]
    if not val_paths:
        # The benchmark disables validation by scheduling it beyond max_iterations,
        # but the training loader still needs a disjoint validation dataset.
        raise RuntimeError("Could not select a validation zarr store disjoint from training paths")

    run_dir = output_dir / run_name
    checkpoint_dir = run_dir / "checkpoints"
    config_path = run_dir / "config.yaml"

    config["loaders"]["batch_size"] = batch_size
    config["loaders"]["num_workers"] = num_workers
    if raw_dtype is not None:
        config["loaders"]["raw_dtype"] = raw_dtype
    else:
        config["loaders"].pop("raw_dtype", None)
    if prefetch_factor is not None:
        config["loaders"]["prefetch_factor"] = prefetch_factor
    else:
        config["loaders"].pop("prefetch_factor", None)
    if multiprocessing_context is not None:
        config["loaders"]["multiprocessing_context"] = multiprocessing_context
    else:
        config["loaders"].pop("multiprocessing_context", None)
    config["loaders"]["train"]["file_paths"] = train_paths
    config["loaders"]["val"]["file_paths"] = val_paths[:1]

    trainer = config["trainer"]
    trainer["checkpoint_dir"] = str(checkpoint_dir)
    trainer["resume"] = None
    if not full_training:
        assert max_iterations is not None
        trainer["max_num_epochs"] = 1_000_000
        trainer["max_num_iterations"] = max_iterations
        trainer["validate_after_iters"] = max_iterations + 10_000
        trainer["log_after_iters"] = max_iterations + 10_000
        trainer["validate_iters"] = 1
        trainer["skip_train_validation"] = True

    _write_yaml(config_path, config)
    return config_path


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
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return children


def _process_tree(root_pid: int) -> list[int]:
    children = _children_by_parent()
    result = []
    stack = [root_pid]
    while stack:
        pid = stack.pop()
        result.append(pid)
        stack.extend(children.get(pid, []))
    return result


def _process_group_pids(pgid: int) -> list[int]:
    pids = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().split()
            if int(fields[4]) == pgid:
                pids.append(int(entry.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError, IndexError):
            continue
    return pids


def _terminate_process_group(pgid: int, grace_seconds: float = 10.0) -> None:
    if not _process_group_pids(pgid):
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return

    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        if not _process_group_pids(pgid):
            return
        time.sleep(0.2)

    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _rss_kb(pid: int) -> int:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return 0
    return 0


def _meminfo_kb() -> dict[str, int]:
    keys = {
        "MemTotal",
        "MemFree",
        "MemAvailable",
        "Buffers",
        "Cached",
        "SReclaimable",
        "SUnreclaim",
        "Shmem",
        "SwapCached",
    }
    values = {key: 0 for key in keys}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            name, raw_value = line.split(":", 1)
            if name in values:
                values[name] = int(raw_value.split()[0])
    except (FileNotFoundError, PermissionError):
        pass
    return values


def _cgroup_memory_bytes() -> int | None:
    try:
        path = Path("/proc/self/cgroup")
        for line in path.read_text().splitlines():
            parts = line.split(":")
            if len(parts) == 3 and parts[0] == "0":
                memory_current = Path("/sys/fs/cgroup") / parts[2].lstrip("/") / "memory.current"
                if memory_current.exists():
                    return int(memory_current.read_text().strip())
    except (FileNotFoundError, PermissionError, ValueError):
        return None
    return None


def _gpu_stats() -> dict[str, float | None]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return {"gpu_util_percent": None, "gpu_memory_used_mb": None, "gpu_memory_total_mb": None}

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        return {"gpu_util_percent": None, "gpu_memory_used_mb": None, "gpu_memory_total_mb": None}

    try:
        util, memory_used, memory_total = [float(value.strip()) for value in lines[0].split(",")]
    except ValueError:
        return {"gpu_util_percent": None, "gpu_memory_used_mb": None, "gpu_memory_total_mb": None}

    return {
        "gpu_util_percent": util,
        "gpu_memory_used_mb": memory_used,
        "gpu_memory_total_mb": memory_total,
    }


def monitor_command(
    command: list[str],
    output_dir: Path,
    run_name: str,
    sample_interval: float,
    timeout_seconds: float | None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{run_name}.memory.csv"
    stdout_path = output_dir / f"{run_name}.stdout.log"
    summary_path = output_dir / f"{run_name}.summary.json"

    repo_root = Path(__file__).resolve().parents[1]
    env = {
        "HOME": os.environ.get("HOME", str(Path.home())),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
        "PATH": "/local0/home/lfranz/.local/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin",
    }
    env["PYTHONPATH"] = f"{repo_root}{os.pathsep}{env.get('PYTHONPATH', '')}"

    start = time.monotonic()
    max_rss_kb = 0
    min_mem_available_kb: int | None = None
    max_cgroup_memory_bytes: int | None = None
    max_gpu_memory_used_mb: float | None = None
    max_gpu_util_percent: float | None = None
    sample_count = 0
    with stdout_path.open("w") as stdout, csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "elapsed_s",
                "pid_count",
                "total_rss_mb",
                "max_total_rss_mb",
                "mem_available_mb",
                "mem_free_mb",
                "cached_mb",
                "buffers_mb",
                "slab_mb",
                "shmem_mb",
                "cgroup_memory_mb",
                "max_cgroup_memory_mb",
                "gpu_util_percent",
                "gpu_memory_used_mb",
                "max_gpu_memory_used_mb",
            ],
        )
        writer.writeheader()
        process = subprocess.Popen(
            command,
            stdout=stdout,
            stderr=subprocess.STDOUT,
            cwd=str(repo_root),
            env=env,
            start_new_session=True,
        )
        timed_out = False
        while True:
            tree = _process_tree(process.pid)
            total_rss_kb = sum(_rss_kb(pid) for pid in tree)
            max_rss_kb = max(max_rss_kb, total_rss_kb)
            meminfo = _meminfo_kb()
            min_mem_available_kb = (
                meminfo["MemAvailable"]
                if min_mem_available_kb is None
                else min(min_mem_available_kb, meminfo["MemAvailable"])
            )
            cgroup_memory_bytes = _cgroup_memory_bytes()
            if cgroup_memory_bytes is not None:
                max_cgroup_memory_bytes = (
                    cgroup_memory_bytes
                    if max_cgroup_memory_bytes is None
                    else max(max_cgroup_memory_bytes, cgroup_memory_bytes)
                )
            gpu_stats = _gpu_stats()
            gpu_memory_used_mb = gpu_stats["gpu_memory_used_mb"]
            gpu_util_percent = gpu_stats["gpu_util_percent"]
            if gpu_memory_used_mb is not None:
                max_gpu_memory_used_mb = (
                    gpu_memory_used_mb
                    if max_gpu_memory_used_mb is None
                    else max(max_gpu_memory_used_mb, gpu_memory_used_mb)
                )
            if gpu_util_percent is not None:
                max_gpu_util_percent = (
                    gpu_util_percent
                    if max_gpu_util_percent is None
                    else max(max_gpu_util_percent, gpu_util_percent)
                )
            elapsed = time.monotonic() - start
            writer.writerow(
                {
                    "elapsed_s": f"{elapsed:.3f}",
                    "pid_count": len(tree),
                    "total_rss_mb": f"{total_rss_kb / 1024:.3f}",
                    "max_total_rss_mb": f"{max_rss_kb / 1024:.3f}",
                    "mem_available_mb": f"{meminfo['MemAvailable'] / 1024:.3f}",
                    "mem_free_mb": f"{meminfo['MemFree'] / 1024:.3f}",
                    "cached_mb": f"{meminfo['Cached'] / 1024:.3f}",
                    "buffers_mb": f"{meminfo['Buffers'] / 1024:.3f}",
                    "slab_mb": f"{(meminfo['SReclaimable'] + meminfo['SUnreclaim']) / 1024:.3f}",
                    "shmem_mb": f"{meminfo['Shmem'] / 1024:.3f}",
                    "cgroup_memory_mb": (
                        "" if cgroup_memory_bytes is None else f"{cgroup_memory_bytes / 1024 / 1024:.3f}"
                    ),
                    "max_cgroup_memory_mb": (
                        "" if max_cgroup_memory_bytes is None else f"{max_cgroup_memory_bytes / 1024 / 1024:.3f}"
                    ),
                    "gpu_util_percent": "" if gpu_util_percent is None else f"{gpu_util_percent:.3f}",
                    "gpu_memory_used_mb": "" if gpu_memory_used_mb is None else f"{gpu_memory_used_mb:.3f}",
                    "max_gpu_memory_used_mb": (
                        "" if max_gpu_memory_used_mb is None else f"{max_gpu_memory_used_mb:.3f}"
                    ),
                }
            )
            csv_file.flush()
            sample_count += 1

            return_code = process.poll()
            if return_code is not None:
                # Some DataLoader worker processes can survive normal trainer exit
                # briefly and keep CUDA/RSS allocated. Clean the whole process group
                # before the next benchmark run starts.
                _terminate_process_group(process.pid)
                break
            if timeout_seconds is not None and elapsed >= timeout_seconds:
                timed_out = True
                _terminate_process_group(process.pid, grace_seconds=30)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                return_code = process.returncode
                break
            time.sleep(sample_interval)

    summary = {
        "run_name": run_name,
        "command": command,
        "return_code": return_code,
        "timed_out": timed_out,
        "elapsed_s": round(time.monotonic() - start, 3),
        "max_rss_mb": round(max_rss_kb / 1024, 3),
        "min_mem_available_mb": None if min_mem_available_kb is None else round(min_mem_available_kb / 1024, 3),
        "max_cgroup_memory_mb": (
            None if max_cgroup_memory_bytes is None else round(max_cgroup_memory_bytes / 1024 / 1024, 3)
        ),
        "max_gpu_memory_used_mb": None if max_gpu_memory_used_mb is None else round(max_gpu_memory_used_mb, 3),
        "max_gpu_util_percent": None if max_gpu_util_percent is None else round(max_gpu_util_percent, 3),
        "samples": sample_count,
        "memory_csv": str(csv_path),
        "stdout_log": str(stdout_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, sort_keys=True))
    return summary


def train_command(config_path: Path) -> list[str]:
    repo_root = Path(__file__).resolve().parents[1]
    return [sys.executable, str(repo_root / "pytorch3dunet" / "train.py"), "--config", str(config_path)]


def cmd_generate(args: argparse.Namespace) -> None:
    config_path = make_config(
        source_config=args.source_config,
        zarr_root=args.zarr_root,
        output_dir=args.output_dir,
        file_count=args.file_count,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_iterations=args.max_iterations,
        run_name=args.run_name,
        raw_dtype=args.raw_dtype,
        prefetch_factor=args.prefetch_factor,
        multiprocessing_context=args.multiprocessing_context,
        full_training=args.full_training,
    )
    print(config_path)


def cmd_run(args: argparse.Namespace) -> None:
    config_path = make_config(
        source_config=args.source_config,
        zarr_root=args.zarr_root,
        output_dir=args.output_dir,
        file_count=args.file_count,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_iterations=args.max_iterations,
        run_name=args.run_name,
        raw_dtype=args.raw_dtype,
        prefetch_factor=args.prefetch_factor,
        multiprocessing_context=args.multiprocessing_context,
        full_training=args.full_training,
    )
    monitor_command(
        train_command(config_path),
        output_dir=args.output_dir / args.run_name,
        run_name=args.run_name,
        sample_interval=args.sample_interval,
        timeout_seconds=None if args.timeout_seconds <= 0 else args.timeout_seconds,
    )


def cmd_summarize(args: argparse.Namespace) -> None:
    rows = []
    for summary_path in sorted(args.output_dir.glob("*/**/*.summary.json")):
        with summary_path.open("r") as f:
            rows.append(json.load(f))

    selected = [row for row in rows if not args.prefix or row["run_name"].startswith(args.prefix)]
    for row in selected:
        print(
            "\t".join(
                [
                    row["run_name"],
                    f"return={row['return_code']}",
                    f"elapsed_s={row['elapsed_s']}",
                    f"max_rss_mb={row['max_rss_mb']}",
                    f"max_cgroup_mb={row.get('max_cgroup_memory_mb')}",
                    f"max_gpu_mem_mb={row.get('max_gpu_memory_used_mb')}",
                    f"min_mem_available_mb={row.get('min_mem_available_mb')}",
                    f"csv={row['memory_csv']}",
                ]
            )
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--source-config", type=Path, default=DEFAULT_SOURCE_CONFIG)
        subparser.add_argument("--zarr-root", type=Path, default=DEFAULT_ZARR_ROOT)
        subparser.add_argument("--output-dir", type=Path, default=DEFAULT_RUNS_DIR)
        subparser.add_argument("--file-count", type=int, default=None)
        subparser.add_argument("--batch-size", type=int, default=15)
        subparser.add_argument("--num-workers", type=int, default=8)
        subparser.add_argument("--max-iterations", type=int, default=None)
        subparser.add_argument("--run-name", required=True)
        subparser.add_argument("--raw-dtype", default=None)
        subparser.add_argument("--prefetch-factor", type=int, default=None)
        subparser.add_argument("--multiprocessing-context", default=None)
        subparser.add_argument("--full-training", action="store_true")

    generate = subparsers.add_parser("generate", help="generate one local training config")
    add_common(generate)
    generate.set_defaults(func=cmd_generate)

    run = subparsers.add_parser("run", help="generate and run one monitored training benchmark")
    add_common(run)
    run.add_argument("--sample-interval", type=float, default=1.0)
    run.add_argument("--timeout-seconds", type=float, default=1800.0)
    run.set_defaults(func=cmd_run)

    summarize = subparsers.add_parser("summarize", help="summarize benchmark result JSON files")
    summarize.add_argument("--output-dir", type=Path, default=DEFAULT_RUNS_DIR)
    summarize.add_argument("--prefix", default="")
    summarize.set_defaults(func=cmd_summarize)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
