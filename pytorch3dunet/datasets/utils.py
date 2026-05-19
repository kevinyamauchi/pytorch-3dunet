import importlib
import os
from collections.abc import Sequence

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, ConcatDataset, Dataset

from pytorch3dunet.unet3d.utils import get_logger

logger = get_logger('Dataset')


class ConfigDataset(Dataset):
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    @classmethod
    def create_datasets(cls, dataset_config, phase):
        """
        Factory method for creating a list of datasets based on the provided config.

        Args:
            dataset_config (dict): dataset configuration
            phase (str): one of ['train', 'val', 'test']

        Returns:
            list of `Dataset` instances
        """
        raise NotImplementedError

    @classmethod
    def iter_test_datasets(cls, dataset_config):
        """
        Generator-style factory yielding one ``phase='test'`` dataset at a time.

        Predictors iterate files sequentially, so there's no need to keep every dataset (and its
        backing handles / slice index / optional mirror-padded raw volume) alive in memory at the
        same time. Subclasses that read from files should override this to construct datasets
        lazily one-by-one.

        The default implementation falls back to :meth:`create_datasets` for backwards
        compatibility (eager, full list kept in memory).
        """
        yield from cls.create_datasets(dataset_config, phase='test')

    @classmethod
    def prediction_collate(cls, batch):
        """Default collate_fn. Override in child class for non-standard datasets."""
        return default_prediction_collate(batch)


class VolumeFileDataset(ConfigDataset):
    """File-backed volumetric patch datasets (HDF5, Zarr, …) used by predictors and loaders."""


class SliceBuilder:
    """
    Builds the position of the patches in a given raw/label/weight ndarray based on the the patch and stride shape
    """

    def __init__(self, raw_datasets, label_datasets, weight_dataset, patch_shape, stride_shape, **kwargs):
        """
        :param raw_datasets: ndarray of raw data
        :param label_datasets: ndarray of ground truth labels
        :param weight_dataset: ndarray of weights for the labels
        :param patch_shape: the shape of the patch DxHxW
        :param stride_shape: the shape of the stride DxHxW
        :param kwargs: additional metadata
        """

        patch_shape = tuple(patch_shape)
        stride_shape = tuple(stride_shape)
        skip_shape_check = kwargs.get('skip_shape_check', False)
        if not skip_shape_check:
            self._check_patch_shape(patch_shape)

        self._raw_slices = self._build_slices(raw_datasets[0], patch_shape, stride_shape)
        if label_datasets is None:
            self._label_slices = None
        else:
            # take the first element in the label_datasets to build slices
            self._label_slices = self._build_slices(label_datasets[0], patch_shape, stride_shape)
            assert len(self._raw_slices) == len(self._label_slices)
        if weight_dataset is None:
            self._weight_slices = None
        else:
            self._weight_slices = self._build_slices(weight_dataset[0], patch_shape, stride_shape)
            assert len(self.raw_slices) == len(self._weight_slices)

    @property
    def raw_slices(self):
        return self._raw_slices

    @property
    def label_slices(self):
        return self._label_slices

    @property
    def weight_slices(self):
        return self._weight_slices

    @staticmethod
    def _build_slices(dataset, patch_shape, stride_shape):
        """Iterates over a given n-dim dataset patch-by-patch with a given stride
        and builds an array of slice positions.

        Returns:
            list of slices, i.e.
            [(slice, slice, slice, slice), ...] if len(shape) == 4
            [(slice, slice, slice), ...] if len(shape) == 3
        """
        slices = []
        if dataset.ndim == 4:
            in_channels, i_z, i_y, i_x = dataset.shape
        else:
            i_z, i_y, i_x = dataset.shape

        k_z, k_y, k_x = patch_shape
        s_z, s_y, s_x = stride_shape
        z_steps = SliceBuilder._gen_indices(i_z, k_z, s_z)
        for z in z_steps:
            y_steps = SliceBuilder._gen_indices(i_y, k_y, s_y)
            for y in y_steps:
                x_steps = SliceBuilder._gen_indices(i_x, k_x, s_x)
                for x in x_steps:
                    slice_idx = (
                        slice(z, z + k_z),
                        slice(y, y + k_y),
                        slice(x, x + k_x)
                    )
                    if dataset.ndim == 4:
                        slice_idx = (slice(0, in_channels),) + slice_idx
                    slices.append(slice_idx)
        return slices

    @staticmethod
    def _gen_indices(i, k, s):
        assert i >= k, 'Sample size has to be bigger than the patch size'
        for j in range(0, i - k + 1, s):
            yield j
        if j + k < i:
            yield i - k

    @staticmethod
    def _check_patch_shape(patch_shape):
        assert len(patch_shape) == 3, 'patch_shape must be a 3D tuple'
        assert patch_shape[1] >= 64 and patch_shape[2] >= 64, 'Height and Width must be greater or equal 64'


class FilterSliceBuilder(SliceBuilder):
    """
    Filter patches containing more than `1 - threshold` of ignore_index label
    """

    def __init__(self, raw_datasets, label_datasets, weight_datasets, patch_shape, stride_shape, ignore_index=(0,),
                 threshold=0.6, slack_acceptance=0.01, **kwargs):
        super().__init__(raw_datasets, label_datasets, weight_datasets, patch_shape, stride_shape, **kwargs)
        if label_datasets is None:
            return

        rand_state = np.random.RandomState(47)

        def ignore_predicate(raw_label_idx):
            label_idx = raw_label_idx[1]
            patch = np.copy(label_datasets[0][label_idx])
            for ii in ignore_index:
                patch[patch == ii] = 0
            non_ignore_counts = np.count_nonzero(patch != 0)
            non_ignore_counts = non_ignore_counts / patch.size
            return non_ignore_counts > threshold or rand_state.rand() < slack_acceptance

        zipped_slices = zip(self.raw_slices, self.label_slices)
        # ignore slices containing too much ignore_index
        filtered_slices = list(filter(ignore_predicate, zipped_slices))
        # unzip and save slices
        raw_slices, label_slices = zip(*filtered_slices)
        self._raw_slices = list(raw_slices)
        self._label_slices = list(label_slices)


class EmbeddingsSliceBuilder(FilterSliceBuilder):
    """
    Filter patches containing more than `1 - threshold` of ignore_index label and patches containing more than
    `patch_max_instances` labels
    """

    def __init__(self, raw_datasets, label_datasets, weight_datasets, patch_shape, stride_shape, ignore_index=(0,),
                 threshold=0.8, slack_acceptance=0.01, patch_max_instances=48, patch_min_instances=5, **kwargs):
        super().__init__(raw_datasets, label_datasets, weight_datasets, patch_shape, stride_shape, ignore_index,
                         threshold, slack_acceptance, **kwargs)

        if label_datasets is None:
            return

        rand_state = np.random.RandomState(47)

        def ignore_predicate(raw_label_idx):
            label_idx = raw_label_idx[1]
            patch = label_datasets[0][label_idx]
            num_instances = np.unique(patch).size

            # patch_max_instances is a hard constraint
            if num_instances <= patch_max_instances:
                # make sure that we have at least patch_min_instances in the batch and allow some slack
                return num_instances >= patch_min_instances or rand_state.rand() < slack_acceptance

            return False

        zipped_slices = zip(self.raw_slices, self.label_slices)
        # ignore slices containing too much ignore_index
        filtered_slices = list(filter(ignore_predicate, zipped_slices))
        # unzip and save slices
        raw_slices, label_slices = zip(*filtered_slices)
        self._raw_slices = list(raw_slices)
        self._label_slices = list(label_slices)


class RandomFilterSliceBuilder(EmbeddingsSliceBuilder):
    """
    Filter patches containing more than `1 - threshold` of ignore_index label and return only random sample of those.
    """

    def __init__(self, raw_datasets, label_datasets, weight_datasets, patch_shape, stride_shape, ignore_index=(0,),
                 threshold=0.8, slack_acceptance=0.01, patch_max_instances=48, patch_acceptance_probab=0.1,
                 max_num_patches=25, **kwargs):
        super().__init__(raw_datasets, label_datasets, weight_datasets, patch_shape, stride_shape,
                         ignore_index=ignore_index, threshold=threshold, slack_acceptance=slack_acceptance,
                         patch_max_instances=patch_max_instances, **kwargs)

        self.max_num_patches = max_num_patches

        if label_datasets is None:
            return

        rand_state = np.random.RandomState(47)

        def ignore_predicate(raw_label_idx):
            result = rand_state.rand() < patch_acceptance_probab
            if result:
                self.max_num_patches -= 1

            return result and self.max_num_patches > 0

        zipped_slices = zip(self.raw_slices, self.label_slices)
        # ignore slices containing too much ignore_index
        filtered_slices = list(filter(ignore_predicate, zipped_slices))
        # unzip and save slices
        raw_slices, label_slices = zip(*filtered_slices)
        self._raw_slices = list(raw_slices)
        self._label_slices = list(label_slices)


def get_class(class_name, modules):
    for module in modules:
        m = importlib.import_module(module)
        clazz = getattr(m, class_name, None)
        if clazz is not None:
            return clazz
    raise RuntimeError(f'Unsupported dataset class: {class_name}')


def _loader_classes(class_name):
    modules = [
        'pytorch3dunet.datasets.hdf5',
        'pytorch3dunet.datasets.zarr_dataset',
        'pytorch3dunet.datasets.memory',
        'pytorch3dunet.datasets.dsb',
        'pytorch3dunet.datasets.utils'
    ]
    return get_class(class_name, modules)


def get_slice_builder(raws, labels, weight_maps, config):
    assert 'name' in config
    logger.info(f"Slice builder config: {config}")
    slice_builder_cls = _loader_classes(config['name'])
    return slice_builder_cls(raws, labels, weight_maps, **config)


def get_train_loaders(config):
    """
    Returns dictionary containing the training and validation loaders (torch.utils.data.DataLoader).

    :param config: a top level configuration object containing the 'loaders' key
    :return: dict {
        'train': <train_loader>
        'val': <val_loader>
    }
    """
    assert 'loaders' in config, 'Could not find data loaders configuration'
    loaders_config = config['loaders']

    logger.info('Creating training and validation set loaders...')

    # get dataset class
    dataset_cls_str = loaders_config.get('dataset', None)
    if dataset_cls_str is None:
        dataset_cls_str = 'StandardHDF5Dataset'
        logger.warn(f"Cannot find dataset class in the config. Using default '{dataset_cls_str}'.")
    dataset_class = _loader_classes(dataset_cls_str)

    assert set(loaders_config['train']['file_paths']).isdisjoint(loaders_config['val']['file_paths']), \
        "Train and validation 'file_paths' overlap. One cannot use validation data for training!"

    train_datasets = dataset_class.create_datasets(loaders_config, phase='train')

    val_datasets = dataset_class.create_datasets(loaders_config, phase='val')

    num_workers = loaders_config.get('num_workers', 1)
    logger.info(f'Number of workers for train/val dataloader: {num_workers}')
    batch_size = loaders_config.get('batch_size', 1)
    if torch.cuda.device_count() > 1 and not config['device'].type == 'cpu':
        logger.info(
            f'{torch.cuda.device_count()} GPUs available. Using batch_size = {torch.cuda.device_count()} * {batch_size}')
        batch_size = batch_size * torch.cuda.device_count()

    logger.info(f'Batch size for train/val loader: {batch_size}')
    # when training with volumetric data use batch_size of 1 due to GPU memory constraints
    return {
        'train': DataLoader(ConcatDataset(train_datasets), batch_size=batch_size, shuffle=True,
                            num_workers=num_workers),
        # don't shuffle during validation: useful when showing how predictions for a given batch get better over time
        'val': DataLoader(ConcatDataset(val_datasets), batch_size=batch_size, shuffle=False, num_workers=num_workers)
    }


def get_test_loaders(config, raw_dataset=None):
    """
    Returns test DataLoader.

    Args:
        config: Dict - Configuration Dictionary
        raw_dataset: List[np.array] Collection of datasets

    Returns:
        generator of DataLoader objects
    """

    assert 'loaders' in config, 'Could not find data loaders configuration'
    loaders_config = config['loaders']

    logger.info('Creating test set loaders...')

    # get dataset class
    dataset_cls_str = loaders_config.get('dataset', None)
    if dataset_cls_str is None:
        dataset_cls_str = 'StandardHDF5Dataset'
        logger.warn(f"Cannot find dataset class in the config. Using default '{dataset_cls_str}'.")
    dataset_class = _loader_classes(dataset_cls_str)

    if dataset_cls_str == "MemoryDataset" and raw_dataset is not None:
        logger.info('Creating datasets from memory.')
        test_datasets = dataset_class.create_datasets(loaders_config,
                                                      raw_dataset=raw_dataset,
                                                      phase='test')
    else:
        logger.info('Creating datasets from files (lazy per-file construction).')
        # iter_test_datasets yields one dataset at a time so that padded raw volumes, slice
        # indexes and open file handles for already-processed files are released before the
        # next file is opened. This bounds test-time memory by a single dataset instead of
        # scaling with the number of input files.
        test_datasets = dataset_class.iter_test_datasets(loaders_config)

    num_workers = loaders_config.get('num_workers', 1)
    logger.info(f'Number of workers for the dataloader: {num_workers}')

    batch_size = loaders_config.get('batch_size', 1)
    if torch.cuda.device_count() > 1 and not config['device'].type == 'cpu':
        logger.info(
            f'{torch.cuda.device_count()} GPUs available. Using batch_size = {torch.cuda.device_count()} * {batch_size}')
        batch_size = batch_size * torch.cuda.device_count()

    logger.info(f'Batch size for dataloader: {batch_size}')

    # use generator in order to create data loaders lazily one by one
    for test_dataset in test_datasets:
        logger.info(f'Loading test set from: {test_dataset.file_path}...')
        if hasattr(test_dataset, 'prediction_collate'):
            collate_fn = test_dataset.prediction_collate
        else:
            collate_fn = default_prediction_collate

        yield DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers,
                         collate_fn=collate_fn)


def default_prediction_collate(batch):
    """
    Default collate_fn to form a mini-batch of Tensor(s) for HDF5 based datasets
    """
    error_msg = "batch must contain tensors or slice; found {}"
    if isinstance(batch[0], torch.Tensor):
        return torch.stack(batch, 0)
    elif isinstance(batch[0], tuple) and isinstance(batch[0][0], slice):
        return batch
    elif isinstance(batch[0], Sequence):
        transposed = zip(*batch)
        return [default_prediction_collate(samples) for samples in transposed]

    raise TypeError((error_msg.format(type(batch[0]))))


def _iter_stats_chunks(image, max_elements=16 * 1024 * 1024):
    if isinstance(image, np.ndarray):
        yield image
        return

    shape = getattr(image, 'shape', None)
    ndim = getattr(image, 'ndim', None)
    if shape is None or ndim is None or ndim < 3:
        yield image[...]
        return

    if ndim == 4:
        channels, _, height, width = shape
        elements_per_z = channels * height * width
        z_axis = 1
    else:
        _, height, width = shape
        elements_per_z = height * width
        z_axis = 0

    step = max(1, max_elements // elements_per_z)
    for z_start in range(0, shape[z_axis], step):
        z_stop = min(shape[z_axis], z_start + step)
        if ndim == 4:
            yield image[(slice(None), slice(z_start, z_stop), slice(None), slice(None))]
        else:
            yield image[(slice(z_start, z_stop), slice(None), slice(None))]


def calculate_stats(images, channelwise=False):
    """
    Calculates min, max, mean, std given a list of ndarrays
    """
    if channelwise:
        return _calculate_channelwise_stats(images)

    return _calculate_global_stats(images)


def _calculate_global_stats(images):
    min_values = np.inf
    max_values = -np.inf
    totals = 0
    sums = 0.0
    sums_sq = 0.0

    for image in images:
        for chunk in _iter_stats_chunks(image):
            chunk = np.asarray(chunk, dtype=np.float64)
            if chunk.size == 0:
                continue

            min_values = min(min_values, np.min(chunk))
            max_values = max(max_values, np.max(chunk))
            totals += chunk.size
            sums += np.sum(chunk)
            sums_sq += np.sum(chunk * chunk)

    if totals == 0:
        raise ValueError('Cannot calculate stats for an empty dataset')

    mean = sums / totals
    variance = max(sums_sq / totals - mean * mean, 0.0)
    return min_values, max_values, mean, np.sqrt(variance)


def _calculate_channelwise_stats(images):
    min_values = None
    max_values = None
    totals = None
    sums = None
    sums_sq = None

    for image in images:
        if getattr(image, 'ndim', None) != 4:
            raise ValueError('Channelwise stats require raw datasets with shape CxDxHxW')

        for chunk in _iter_stats_chunks(image):
            chunk = np.asarray(chunk, dtype=np.float64)
            if chunk.size == 0:
                continue

            axes = tuple(range(1, chunk.ndim))
            chunk_min = np.min(chunk, axis=axes)
            chunk_max = np.max(chunk, axis=axes)
            chunk_sum = np.sum(chunk, axis=axes)
            chunk_sum_sq = np.sum(chunk * chunk, axis=axes)
            chunk_total = np.prod(chunk.shape[1:])

            if min_values is None:
                min_values = chunk_min
                max_values = chunk_max
                totals = np.zeros(chunk.shape[0], dtype=np.float64)
                sums = np.zeros(chunk.shape[0], dtype=np.float64)
                sums_sq = np.zeros(chunk.shape[0], dtype=np.float64)
            else:
                min_values = np.minimum(min_values, chunk_min)
                max_values = np.maximum(max_values, chunk_max)

            totals += chunk_total
            sums += chunk_sum
            sums_sq += chunk_sum_sq

    if totals is None or np.any(totals == 0):
        raise ValueError('Cannot calculate stats for an empty dataset')

    mean = sums / totals
    variance = np.maximum(sums_sq / totals - mean * mean, 0.0)
    return min_values.tolist(), max_values.tolist(), mean.tolist(), np.sqrt(variance).tolist()


def load_stats_from_yaml(stats_file, file_path):
    """
    Load per-file min/max/mean/std values from YAML.

    Supported layouts:
      /path/to/file.h5: {min: 0, max: 1, mean: 0.5, std: 0.1}
      files:
        - path: /path/to/file.h5
          min: 0
          max: 1
          mean: 0.5
          std: 0.1
    """
    with open(stats_file, 'r') as f:
        stats_config = yaml.load(f, Loader=yaml.SafeLoader)

    entry = _find_stats_entry(stats_config, file_path)
    if entry is None:
        raise KeyError(f'Cannot find stats for {file_path} in {stats_file}')

    return _read_stats_values(entry, stats_file, file_path)


def get_ds_stats_file(dataset_config):
    stats_config = dataset_config.get('ds_stats_file', dataset_config.get('stats_file', None))
    if isinstance(stats_config, dict):
        return stats_config.get('path', stats_config.get('file', None))
    return stats_config


def _find_stats_entry(stats_config, file_path):
    if stats_config is None:
        return None

    if _has_stats_keys(stats_config):
        return stats_config

    files = stats_config.get('files') if isinstance(stats_config, dict) else None
    if isinstance(files, dict):
        entry = _find_stats_entry(files, file_path)
        if entry is not None:
            return entry
    elif isinstance(files, list):
        for item in files:
            if not isinstance(item, dict):
                continue
            if _matches_file_path(item.get('path') or item.get('file_path') or item.get('file'), file_path):
                return item.get('stats', item)

    if not isinstance(stats_config, dict):
        return None

    for key in _file_path_keys(file_path):
        if key in stats_config:
            entry = stats_config[key]
            return entry.get('stats', entry) if isinstance(entry, dict) else entry

    return None


def _read_stats_values(entry, stats_file, file_path):
    if not isinstance(entry, dict):
        raise ValueError(f'Invalid stats entry for {file_path} in {stats_file}: expected a mapping')

    min_value = entry.get('min', entry.get('min_value'))
    max_value = entry.get('max', entry.get('max_value'))
    mean = entry.get('mean')
    std = entry.get('std')

    missing = [name for name, value in [('min', min_value), ('max', max_value), ('mean', mean), ('std', std)]
               if value is None]
    if missing:
        raise KeyError(f'Missing stats keys for {file_path} in {stats_file}: {missing}')

    return _stats_value(min_value), _stats_value(max_value), _stats_value(mean), _stats_value(std)


def _stats_value(value):
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    return float(value)


def _has_stats_keys(entry):
    if not isinstance(entry, dict):
        return False
    has_min = 'min' in entry or 'min_value' in entry
    has_max = 'max' in entry or 'max_value' in entry
    return has_min and has_max and 'mean' in entry and 'std' in entry


def _matches_file_path(candidate, file_path):
    if candidate is None:
        return False
    return str(candidate) in _file_path_keys(file_path)


def _file_path_keys(file_path):
    return {
        str(file_path),
        os.path.abspath(file_path),
        os.path.realpath(file_path),
        os.path.basename(file_path),
    }


def sample_instances(label_img, instance_ratio, random_state, ignore_labels=(0,)):
    """
    Given the labelled volume `label_img`, this function takes a random subset of object instances specified by `instance_ratio`
    and zeros out the remaining labels.

    Args:
        label_img(nd.array): labelled image
        instance_ratio(float): a number from (0, 1]
        random_state: RNG state
        ignore_labels: labels to be ignored during sampling

    Returns:
         labelled volume of the same size as `label_img` with a random subset of object instances.
    """
    unique = np.unique(label_img)
    for il in ignore_labels:
        unique = np.setdiff1d(unique, il)

    # shuffle labels
    random_state.shuffle(unique)
    # pick instance_ratio objects
    num_objects = round(instance_ratio * len(unique))
    if num_objects == 0:
        # if there are no objects left, just return an empty patch
        return np.zeros_like(label_img)

    # sample the labels
    sampled_instances = unique[:num_objects]

    result = np.zeros_like(label_img)
    # keep only the sampled_instances
    for si in sampled_instances:
        result[label_img == si] = si

    return result
