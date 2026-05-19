#!/usr/bin/env python3
import argparse
from concurrent.futures import ProcessPoolExecutor
import os
import sys
from pathlib import Path

import h5py
import yaml
import zarr

from pytorch3dunet.datasets.utils import calculate_stats


DEFAULT_INPUT_DIR = '/local1/lfranz/local_data/validated_raw/BiCycle/combined_zarr/'
DEFAULT_OUTPUT = '/local1/lfranz/local_data/validated_raw/BiCycle/dataset_stats.yaml'
DEFAULT_INTERNAL_PATH = 'autocontext_rescaled_BiCycle_fold_0'
HDF5_EXTENSIONS = ('.h5', '.hdf5', '.hd5', '.hdf')


def main():
    parser = argparse.ArgumentParser(
        description='Calculate per-file min/max/mean/std for Zarr or HDF5 volumes and save them as YAML.'
    )
    parser.add_argument('inputs', nargs='*',
                        help='Zarr/HDF5 files or directories to scan. Defaults to --input-dir when omitted.')
    parser.add_argument('--input-dir', default=DEFAULT_INPUT_DIR,
                        help='Directory to scan when no positional inputs are provided')
    parser.add_argument('--output', default=DEFAULT_OUTPUT, help='Output YAML path')
    parser.add_argument('--internal-path', default=DEFAULT_INTERNAL_PATH,
                        help='Dataset path inside each Zarr store or HDF5 file')
    parser.add_argument('--global-stats', action='store_true',
                        help='Calculate one value across all channels instead of per-channel values')
    parser.add_argument('--strict', action='store_true',
                        help='Fail immediately when a volume cannot be processed')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of worker processes. Defaults to ProcessPoolExecutor default.')
    args = parser.parse_args()

    input_roots = args.inputs or [args.input_dir]
    volume_paths = iter_volume_paths(input_roots)
    if not volume_paths:
        raise RuntimeError(f'No Zarr or HDF5 volumes found in: {input_roots}')

    stats = {'files': {}}
    skipped = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Convert Path objects to strings for serialization safety in subprocesses.
        jobs = [(str(p), args.internal_path, not args.global_stats) for p in volume_paths]
        for abs_path, file_stats, error in executor.map(process_volume_file, jobs):
            if error is not None:
                if args.strict:
                    raise RuntimeError(error)
                skipped.append((abs_path, error))
                print(f'Skipping {abs_path}: {error}', file=sys.stderr, flush=True)
                continue
            stats['files'][abs_path] = file_stats

    if not stats['files']:
        raise RuntimeError('No volumes were processed successfully')

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w') as f:
        yaml.safe_dump(stats, f, sort_keys=True)

    print(f'Wrote stats for {len(stats["files"])} volumes to {output_path}', flush=True)
    if skipped:
        print(f'Skipped {len(skipped)} volumes; processed {len(stats["files"])} successfully', flush=True)


def iter_volume_paths(roots):
    paths = []
    for root in roots:
        root_path = Path(root)
        if is_zarr_path(root_path) or is_hdf5_path(root_path):
            paths.append(root_path)
        elif root_path.is_dir():
            paths.extend(iter_volume_paths_in_dir(root_path))
        else:
            raise FileNotFoundError(root)
    return sorted(set(paths))


def iter_volume_paths_in_dir(root_path):
    paths = []
    for dirpath, dirnames, filenames in os.walk(root_path):
        for dirname in list(dirnames):
            path = Path(dirpath) / dirname
            if is_zarr_path(path):
                paths.append(path)
                dirnames.remove(dirname)

        for filename in filenames:
            path = Path(dirpath) / filename
            if is_hdf5_path(path):
                paths.append(path)

    return paths


def is_zarr_path(path):
    return path.is_dir() and path.name.endswith('.zarr')


def is_hdf5_path(path):
    return path.is_file() and path.suffix.lower() in HDF5_EXTENSIONS


def process_volume_file(job):
    volume_path_str, internal_path, channelwise = job
    abs_path = os.path.abspath(volume_path_str)

    try:
        print(f'Calculating stats for {volume_path_str}', flush=True)
        if volume_path_str.endswith('.zarr'):
            root = zarr.open_group(volume_path_str, mode='r')
            return calculate_volume_stats(abs_path, root, internal_path, channelwise)

        with h5py.File(volume_path_str, 'r') as root:
            return calculate_volume_stats(abs_path, root, internal_path, channelwise)
    except Exception as exc:
        return abs_path, None, f'{type(exc).__name__}: {exc}'


def calculate_volume_stats(abs_path, root, internal_path, channelwise):
    try:
        raw = root[internal_path]
    except KeyError:
        available = available_arrays(root)
        details = f'missing internal path {internal_path!r}'
        if available:
            details = f'{details}; available arrays: {", ".join(available)}'
        return abs_path, None, details

    min_value, max_value, mean, std = calculate_stats([raw], channelwise=channelwise)
    return (
        abs_path,
        {
            'min': _to_builtin(min_value),
            'max': _to_builtin(max_value),
            'mean': _to_builtin(mean),
            'std': _to_builtin(std),
        },
        None,
    )


def available_arrays(group, prefix=''):
    names = []
    for key in group.keys():
        try:
            value = group[key]
        except Exception:
            continue
        name = f'{prefix}/{key}' if prefix else key
        if hasattr(value, 'shape'):
            names.append(name)
        elif hasattr(value, 'keys'):
            names.extend(available_arrays(value, name))
    return names


def _to_builtin(value):
    if isinstance(value, list):
        return [_to_builtin(v) for v in value]
    if hasattr(value, 'item'):
        return value.item()
    return value


if __name__ == '__main__':
    main()
