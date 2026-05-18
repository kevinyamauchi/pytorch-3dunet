#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import concurrent.futures
import yaml
import zarr

from pytorch3dunet.datasets.utils import calculate_stats


DEFAULT_INPUT_DIR = '/local1/lfranz/local_data/validated_raw/BiCycle/combined_zarr/'
DEFAULT_OUTPUT = '/local1/lfranz/local_data/validated_raw/BiCycle/dataset_stats.yaml'
DEFAULT_INTERNAL_PATH = 'autocontext_rescaled_BiCycle_fold_0'


def main():
    parser = argparse.ArgumentParser(
        description='Calculate per-file min/max/mean/std for Zarr volumes and save them as YAML.'
    )
    parser.add_argument('--input-dir', default=DEFAULT_INPUT_DIR, help='Directory containing *.zarr stores')
    parser.add_argument('--output', default=DEFAULT_OUTPUT, help='Output YAML path')
    parser.add_argument('--internal-path', default=DEFAULT_INTERNAL_PATH, help='Dataset path inside each Zarr store')
    parser.add_argument('--global-stats', action='store_true',
                        help='Calculate one value across all channels instead of per-channel values')
    args = parser.parse_args()

    zarr_paths = sorted(Path(args.input_dir).glob('*.zarr'))
    if not zarr_paths:
        raise RuntimeError(f'No .zarr stores found in {args.input_dir}')

    stats = {'files': {}}
    

    def process_zarr_file(zarr_path_str):
        print(f'Calculating stats for {zarr_path_str}', flush=True)
        root = zarr.open_group(str(zarr_path_str), mode='r')
        raw = root[args.internal_path]
        min_value, max_value, mean, std = calculate_stats([raw], channelwise=not args.global_stats)
        return (
            os.path.abspath(zarr_path_str),
            {
                'min': _to_builtin(min_value),
                'max': _to_builtin(max_value),
                'mean': _to_builtin(mean),
                'std': _to_builtin(std),
            }
        )

    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Convert Path objects to strings for serialization safety in subprocesses.
        zarr_path_strs = [str(p) for p in zarr_paths]
        results = list(executor.map(process_zarr_file, zarr_path_strs))
        for abs_path, file_stats in results:
            stats['files'][abs_path] = file_stats
 

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w') as f:
        yaml.safe_dump(stats, f, sort_keys=True)

    print(f'Wrote stats for {len(zarr_paths)} volumes to {output_path}', flush=True)


def _to_builtin(value):
    if isinstance(value, list):
        return [_to_builtin(v) for v in value]
    if hasattr(value, 'item'):
        return value.item()
    return value


if __name__ == '__main__':
    main()
