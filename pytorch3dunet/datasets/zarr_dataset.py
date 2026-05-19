"""
Zarr-backed volumetric datasets for parallel-friendly lazy I/O.

Each training/validation file is a directory store ending in ``.zarr`` (Zarr group) containing the same
dataset names as in the original HDF5 (e.g. ``raw_rescaled``, ``lsds``). Unlike ``LazyHDF5Dataset``, multiple
``DataLoader`` workers are safe as long as each worker opens its own store (the default PyTorch worker model).
"""
import glob
import os
from itertools import chain

import numpy as np
import zarr

import pytorch3dunet.augment.transforms as transforms
from pytorch3dunet.datasets.utils import (
    calculate_stats,
    get_ds_stats_file,
    get_slice_builder,
    load_stats_from_yaml,
    sample_instances,
    VolumeFileDataset,
)
from pytorch3dunet.unet3d.utils import get_logger

logger = get_logger('ZarrDataset')


class AbstractZarrDataset(VolumeFileDataset):
    """
    Same patch iteration contract as ``AbstractHDF5Dataset``, but arrays are Zarr (lazy until sliced).
    """

    def __init__(self, file_path,
                 phase,
                 slice_builder_config,
                 transformer_config,
                 mirror_padding=(16, 32, 32),
                 raw_internal_path='raw',
                 label_internal_path='label',
                 weight_internal_path=None,
                 instance_ratio=None,
                 random_seed=0,
                 stats_file=None):
        assert phase in ['train', 'val', 'test']
        if phase in ['train', 'val']:
            mirror_padding = None

        if mirror_padding is not None:
            if isinstance(mirror_padding, int):
                mirror_padding = (mirror_padding,) * 3
            else:
                assert len(mirror_padding) == 3, f"Invalid mirror_padding: {mirror_padding}"

        self.mirror_padding = mirror_padding
        self.phase = phase
        self.file_path = file_path
        self.stats_file = stats_file
        self.transformer_config = transformer_config

        self.instance_ratio = instance_ratio

        if isinstance(raw_internal_path, str):
            raw_internal_path = [raw_internal_path]
        if isinstance(label_internal_path, str):
            label_internal_path = [label_internal_path]
        if isinstance(weight_internal_path, str):
            weight_internal_path = [weight_internal_path]

        internal_paths = list(raw_internal_path)
        if label_internal_path is not None:
            internal_paths.extend(label_internal_path)
        if weight_internal_path is not None:
            internal_paths.extend(weight_internal_path)

        root = self.open_zarr_group(file_path, internal_paths)

        self.raws = self.fetch_and_check(root, raw_internal_path)

        min_value, max_value, mean, std = self.ds_stats()

        self.transformer = transforms.get_transformer(transformer_config, min_value=min_value, max_value=max_value,
                                                      mean=mean, std=std)
        self.raw_transform = self.transformer.raw_transform()

        if phase != 'test':
            self.label_transform = self.transformer.label_transform()
            self.labels = self.fetch_and_check(root, label_internal_path)

            if self.instance_ratio is not None:
                assert 0 < self.instance_ratio <= 1
                rs = np.random.RandomState(random_seed)
                self.labels = [sample_instances(m, self.instance_ratio, rs) for m in self.labels]

            if weight_internal_path is not None:
                self.weight_maps = self.fetch_and_check(root, weight_internal_path)
                self.weight_transform = self.transformer.weight_transform()
            else:
                self.weight_maps = None

            self._check_dimensionality(self.raws, self.labels)
        else:
            self.labels = None
            self.weight_maps = None

            if self.mirror_padding is not None:
                z, y, x = self.mirror_padding
                pad_width = ((z, z), (y, y), (x, x))
                padded_volumes = []
                for raw in self.raws:
                    if raw.ndim == 4:
                        channels = [np.pad(r, pad_width=pad_width, mode='reflect') for r in raw]
                        padded_volume = np.stack(channels)
                    else:
                        padded_volume = np.pad(raw, pad_width=pad_width, mode='reflect')

                    padded_volumes.append(padded_volume)

                self.raws = padded_volumes

        slice_builder = get_slice_builder(self.raws, self.labels, self.weight_maps, slice_builder_config)
        self.raw_slices = slice_builder.raw_slices
        self.label_slices = slice_builder.label_slices
        self.weight_slices = slice_builder.weight_slices

        self.patch_count = len(self.raw_slices)
        logger.info(f'Number of patches: {self.patch_count}')

    def ds_stats(self):
        min_value, max_value, mean, std = self._load_or_calculate_stats()
        logger.info(f'Input stats: min={min_value}, max={max_value}, mean={mean}, std={std}')
        return min_value, max_value, mean, std

    def _load_or_calculate_stats(self):
        if self.stats_file is not None:
            logger.info(f'Loading input stats for {self.file_path} from {self.stats_file}')
            return load_stats_from_yaml(self.stats_file, self.file_path)

        return calculate_stats(self.raws, channelwise=_standardize_channelwise(self.transformer_config))

    @staticmethod
    def open_zarr_group(file_path, internal_paths):
        raise NotImplementedError

    @staticmethod
    def fetch_datasets(root, internal_paths):
        raise NotImplementedError

    def fetch_and_check(self, root, internal_paths):
        datasets = self.fetch_datasets(root, internal_paths)
        fn = lambda ds: np.expand_dims(ds, axis=0) if ds.ndim == 2 else ds
        datasets = list(map(fn, datasets))
        return datasets

    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration

        raw_idx = self.raw_slices[idx]
        raw_patch_transformed = self._transform_patches(self.raws, raw_idx, self.raw_transform)

        if self.phase == 'test':
            if len(raw_idx) == 4:
                raw_idx = raw_idx[1:]
            return raw_patch_transformed, raw_idx
        else:
            label_idx = self.label_slices[idx]
            label_patch_transformed = self._transform_patches(self.labels, label_idx, self.label_transform)
            if self.weight_maps is not None:
                weight_idx = self.weight_slices[idx]
                weight_patch_transformed = self._transform_patches(self.weight_maps, weight_idx, self.weight_transform)
                return raw_patch_transformed, label_patch_transformed, weight_patch_transformed
            return raw_patch_transformed, label_patch_transformed

    @staticmethod
    def _transform_patches(datasets, label_idx, transformer):
        transformed_patches = []
        for dataset in datasets:
            transformed_patch = transformer(dataset[label_idx])
            transformed_patches.append(transformed_patch)

        if len(transformed_patches) == 1:
            return transformed_patches[0]
        else:
            return transformed_patches

    def __len__(self):
        return self.patch_count

    @staticmethod
    def _check_dimensionality(raws, labels):
        def _volume_shape(volume):
            if volume.ndim == 3:
                return volume.shape
            return volume.shape[1:]

        for raw, label in zip(raws, labels):
            assert raw.ndim in [3, 4], 'Raw dataset must be 3D (DxHxW) or 4D (CxDxHxW)'
            assert label.ndim in [3, 4], 'Label dataset must be 3D (DxHxW) or 4D (CxDxHxW)'

            assert _volume_shape(raw) == _volume_shape(label), 'Raw and labels have to be of the same size'

    @classmethod
    def _iter_datasets(cls, dataset_config, phase):
        """
        Generator yielding one dataset per file in ``dataset_config[phase]['file_paths']``.
        Failures loading a single file are logged and skipped; iteration continues with the
        next file.
        """
        phase_config = dataset_config[phase]

        transformer_config = phase_config['transformer']
        slice_builder_config = phase_config['slice_builder']
        file_paths = phase_config['file_paths']
        file_paths = cls.traverse_zarr_paths(file_paths)

        instance_ratio = phase_config.get('instance_ratio', None)
        random_seed = phase_config.get('random_seed', 0)

        for file_path in file_paths:
            try:
                logger.info(f'Loading {phase} set from: {file_path}...')
                dataset = cls(file_path=file_path,
                              phase=phase,
                              slice_builder_config=slice_builder_config,
                              transformer_config=transformer_config,
                              mirror_padding=dataset_config.get('mirror_padding', None),
                              raw_internal_path=dataset_config.get('raw_internal_path', 'raw'),
                              label_internal_path=dataset_config.get('label_internal_path', 'label'),
                              weight_internal_path=dataset_config.get('weight_internal_path', None),
                              instance_ratio=instance_ratio,
                              random_seed=random_seed,
                              stats_file=get_ds_stats_file(dataset_config))
            except Exception:
                logger.error(f'Skipping {phase} set: {file_path}', exc_info=True)
                continue
            yield dataset

    @classmethod
    def create_datasets(cls, dataset_config, phase):
        return list(cls._iter_datasets(dataset_config, phase))

    @classmethod
    def iter_test_datasets(cls, dataset_config):
        return cls._iter_datasets(dataset_config, phase='test')

    @staticmethod
    def traverse_zarr_paths(file_paths):
        assert isinstance(file_paths, list)
        results = []
        for file_path in file_paths:
            if os.path.isdir(file_path):
                # Direct .zarr store path: keep it as a dataset root.
                if file_path.endswith('.zarr'):
                    results.append(file_path)
                    continue
                iters = [glob.glob(os.path.join(file_path, ext)) for ext in ['*.zarr']]
                for fp in chain(*iters):
                    results.append(fp)
            else:
                results.append(file_path)
        return results


class LazyZarrDataset(AbstractZarrDataset):
    """
    Lazy Zarr volumes (chunked on disk). Prefer ``num_workers >= 1`` for throughput; each worker process
    opens its own Zarr store.

    Full-volume min/max/mean/std are calculated once at startup, or loaded from ``ds_stats_file``.
    """

    def ds_stats(self):
        min_value, max_value, mean, std = self._load_or_calculate_stats()
        logger.info(f'LazyZarrDataset input stats: min={min_value}, max={max_value}, mean={mean}, std={std}')
        return min_value, max_value, mean, std

    @staticmethod
    def open_zarr_group(file_path, internal_paths):
        if not os.path.isdir(file_path):
            raise FileNotFoundError(f'Zarr store must be a directory: {file_path}')
        return zarr.open_group(file_path, mode='r')

    @staticmethod
    def fetch_datasets(root, internal_paths):
        return [root[internal_path] for internal_path in internal_paths]


class StandardZarrDataset(AbstractZarrDataset):
    """Loads entire Zarr arrays into memory (faster epoch time, high RAM)."""

    @staticmethod
    def open_zarr_group(file_path, internal_paths):
        if not os.path.isdir(file_path):
            raise FileNotFoundError(f'Zarr store must be a directory: {file_path}')
        return zarr.open_group(file_path, mode='r')

    @staticmethod
    def fetch_datasets(root, internal_paths):
        return [np.asarray(root[internal_path]) for internal_path in internal_paths]


def _standardize_channelwise(transformer_config):
    for transform in transformer_config.get('raw', []):
        if transform.get('name') == 'Standardize':
            return transform.get('channelwise', False)
    return False
