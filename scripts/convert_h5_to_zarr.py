#!/usr/bin/env python3
"""
Convert HDF5 volumes to Zarr group stores (one ``.zarr`` directory per input file).

Recursively copies all datasets and sub-groups so configs can keep the same ``raw_internal_path`` /
``label_internal_path`` keys as in the original H5.
"""
import argparse
import os
import sys
from typing import Tuple

import h5py
import numpy as np
import zarr
from zarr.codecs import BloscCodec, BloscShuffle


def _default_chunks(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    if len(shape) == 3:
        z, y, x = shape
        return (min(80, max(1, z)), min(80, y), min(80, x))
    if len(shape) == 4:
        c, z, y, x = shape
        return (1, min(80, max(1, z)), min(80, y), min(80, x))
    if len(shape) == 2:
        return tuple(min(512, s) for s in shape)
    return shape


def _copy_slabwise(src: h5py.Dataset, dst: zarr.Array) -> None:
    shape = src.shape
    if len(shape) == 0:
        dst[()] = np.asarray(src[()])
        return
    lead = shape[0]
    step = min(32, max(1, lead))
    for start in range(0, lead, step):
        end = min(start + step, lead)
        idx = (slice(start, end),) + (slice(None),) * (len(shape) - 1)
        dst[idx] = np.asarray(src[idx])


def _h5_dataset_to_zarr(h5_ds: h5py.Dataset, z_parent, name: str, compressor) -> None:
    shape = h5_ds.shape
    dtype = h5_ds.dtype
    chunks = _default_chunks(shape)
    arr = z_parent.create_array(
        name, shape=shape, dtype=dtype, chunks=chunks, compressors=compressor,
    )
    _copy_slabwise(h5_ds, arr)


def _copy_h5_group_to_zarr(h5_grp: h5py.Group, z_grp, compressor) -> None:
    for key in h5_grp.keys():
        # LazyHDF5Dataset creates "_uncompressed_*" mirrors in H5.
        # They are implementation artifacts and duplicate the true datasets.
        if key.startswith('_uncompressed_'):
            continue
        obj = h5_grp[key]
        if isinstance(obj, h5py.Dataset):
            _h5_dataset_to_zarr(obj, z_grp, key, compressor)
        else:
            sub = z_grp.create_group(key)
            _copy_h5_group_to_zarr(obj, sub, compressor)


def convert_h5_file(h5_path: str, compressor, overwrite: bool = False, delete_h5: bool = False) -> str:
    h5_path = os.path.abspath(h5_path)
    if not os.path.isfile(h5_path):
        raise FileNotFoundError(h5_path)

    root, ext = os.path.splitext(h5_path)
    if ext.lower() not in ('.h5', '.hdf5', '.hd5', '.hdf'):
        raise ValueError(f'Not an HDF5 file extension: {h5_path}')

    zarr_path = root + '.zarr'
    if os.path.exists(zarr_path):
        if not overwrite:
            raise FileExistsError(f'Already exists (use --overwrite): {zarr_path}')
        import shutil

        shutil.rmtree(zarr_path)

    with h5py.File(h5_path, 'r') as h5f:
        z_root = zarr.open_group(zarr_path, mode='w')
        _copy_h5_group_to_zarr(h5f, z_root, compressor)

    if delete_h5:
        os.remove(h5_path)

    return zarr_path


def _iter_h5_files(roots: list[str]) -> list[str]:
    out: list[str] = []
    for root in roots:
        root = os.path.abspath(root)
        if os.path.isfile(root):
            out.append(root)
        elif os.path.isdir(root):
            for dirpath, _, filenames in os.walk(root):
                for fn in filenames:
                    if fn.lower().endswith(('.h5', '.hdf5', '.hd5', '.hdf')):
                        out.append(os.path.join(dirpath, fn))
        else:
            raise FileNotFoundError(root)
    return sorted(set(out))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        'roots',
        nargs='+',
        help='HDF5 files and/or directories to walk for *.h5 / *.hdf5',
    )
    p.add_argument('--overwrite', action='store_true', help='Replace existing .zarr directories')
    p.add_argument('--delete-h5', action='store_true', help='Remove each source .h5 after successful conversion')
    args = p.parse_args()

    compressor = BloscCodec(cname='zstd', clevel=5, shuffle=BloscShuffle.shuffle)
    files = _iter_h5_files(args.roots)
    if not files:
        print('No HDF5 files found.', file=sys.stderr)
        return 1

    for fp in files:
        stem, _ext = os.path.splitext(fp)
        zpath = stem + '.zarr'
        if os.path.exists(zpath) and not args.overwrite:
            print(f'skip (zarr exists): {fp}', flush=True)
            continue
        print(f'convert: {fp} -> {zpath}', flush=True)
        convert_h5_file(fp, compressor, overwrite=args.overwrite, delete_h5=args.delete_h5)

    print('Done.', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
