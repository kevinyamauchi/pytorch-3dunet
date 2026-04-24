from collections.abc import Sequence
import importlib

import numpy as np
import torch
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


class _ReflectPaddedView:
    """
    Lazy numpy-like view of ``inner`` with reflect-mode padding applied on-the-fly per patch.

    Presents a padded ``.shape`` / ``.ndim`` / ``.dtype`` but never materialises the full
    padded volume. When a sub-region is requested via ``__getitem__``, only the intersection
    with the underlying array is read (single slice into the backing zarr/h5 array), and
    reflect padding is applied to that sub-array to fill in the requested region that falls
    outside the original volume.

    This is the key difference to ``np.pad(inner, pad_width, mode='reflect')``, which would
    (a) force a full materialisation of ``inner`` into RAM via ``np.asarray`` and (b) hold
    on to a padded copy. For a typical CxDxHxW prediction volume of ~250 MB that otherwise
    multiplied by the number of test files in a fold (hundreds), this drove OOMs during
    Stage 1 LSD prediction.

    Correctness
    -----------
    Equivalent to ``np.pad(np.asarray(inner), pad_width, mode='reflect')[key]`` for any
    ``key`` produced by the ``SliceBuilder`` (tuple of non-step slices, one per axis).
    The reflect pattern matches because when a patch extends past the left boundary we
    always fetch the inner sub-array starting at index 0 (so the slab edge matches the
    volume edge), and symmetrically for the right boundary. See tests.

    Constraints
    -----------
    * Only slice indexing with ``step`` in ``(None, 1)`` is supported.
    * The backing array must expose ``.shape``, ``.ndim``, ``.dtype`` and ``__getitem__``.
    * Each padded axis must satisfy ``inner_len > max(before, after)`` (same constraint as
      ``numpy.pad(..., mode='reflect')``).
    """

    def __init__(self, inner, pad_width):
        """
        Args:
            inner: array-like (e.g. ``numpy.ndarray``, ``zarr.Array``, ``h5py.Dataset``).
            pad_width: sequence of ``(before, after)`` pairs, one per axis of ``inner``.
                Use ``(0, 0)`` for axes that should not be padded (e.g. the channel axis
                of a CxDxHxW raw array).
        """
        assert hasattr(inner, 'shape') and hasattr(inner, 'ndim') and hasattr(inner, 'dtype'), \
            f'inner must expose shape/ndim/dtype, got {type(inner).__name__}'
        assert len(pad_width) == inner.ndim, \
            f'pad_width length ({len(pad_width)}) must match inner.ndim ({inner.ndim})'

        self._inner = inner
        self._pad_width = tuple((int(b), int(a)) for b, a in pad_width)
        self.ndim = inner.ndim
        self.dtype = inner.dtype
        self.shape = tuple(s + b + a for s, (b, a) in zip(inner.shape, self._pad_width))

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        assert len(key) == self.ndim, \
            f'Expected {self.ndim}-element index, got {len(key)}'

        fetch_slices = []
        reflect_pad = []
        trim_slices = []
        needs_trim = False
        for axis, s in enumerate(key):
            assert isinstance(s, slice), \
                f'Only slice indexing is supported, got {type(s).__name__} on axis {axis}'
            assert s.step in (None, 1), \
                f'Step slicing is not supported (axis {axis}, step {s.step})'

            inner_len = self._inner.shape[axis]
            pad_before, pad_after = self._pad_width[axis]

            start = 0 if s.start is None else s.start
            stop = self.shape[axis] if s.stop is None else s.stop

            i_start = start - pad_before
            i_stop = stop - pad_before

            left_pad = max(0, -i_start)
            right_pad = max(0, i_stop - inner_len)

            if left_pad == 0 and right_pad == 0:
                # Interior patch: simple clamped fetch, no pad, no trim.
                fetch_slices.append(slice(i_start, i_stop))
                reflect_pad.append((0, 0))
                trim_slices.append(slice(None))
                continue

            # Boundary patch on this axis. We over-fetch just enough so that np.pad(..., 'reflect')
            # on the fetched slab produces the same result as np.pad(full_inner_axis, 'reflect')
            # restricted to the requested padded range. Two conditions:
            #   * When left_pad > 0, the slab must start at inner index 0 (so the left edge of
            #     the slab equals the left edge of the full volume). Similarly, when right_pad > 0,
            #     the slab must end at inner index inner_len.
            #   * Additionally, slab_len must be > max(left_pad, right_pad) for numpy reflect to
            #     mirror without "bouncing" onto the opposite edge of the slab (which would not
            #     match the full-volume pattern).
            if left_pad > 0 and right_pad > 0:
                f_start_ext = 0
                f_stop_ext = inner_len
            elif left_pad > 0:
                f_start_ext = 0
                f_stop_ext = max(i_stop, left_pad + 1)
                f_stop_ext = min(f_stop_ext, inner_len)
            else:  # right_pad > 0, left_pad == 0
                f_start_ext = min(i_start, inner_len - right_pad - 1)
                f_start_ext = max(0, f_start_ext)
                f_stop_ext = inner_len

            # After reflect padding the slab represents padded indices
            # [pad_before + f_start_ext - left_pad, pad_before + f_stop_ext + right_pad).
            # We want slab-local indices that map back to the requested padded range [start, stop).
            trim_start = (start - pad_before) - f_start_ext + left_pad
            trim_stop = trim_start + (stop - start)

            fetch_slices.append(slice(f_start_ext, f_stop_ext))
            reflect_pad.append((left_pad, right_pad))
            trim_slices.append(slice(trim_start, trim_stop))
            needs_trim = True

        inner_patch = np.asarray(self._inner[tuple(fetch_slices)])
        if any(l or r for l, r in reflect_pad):
            inner_patch = np.pad(inner_patch, reflect_pad, mode='reflect')
        if needs_trim:
            inner_patch = inner_patch[tuple(trim_slices)]
        return inner_patch


def apply_lazy_mirror_padding(raws, mirror_padding):
    """
    Wrap each 3D (DxHxW) or 4D (CxDxHxW) raw array in a ``_ReflectPaddedView`` applying
    ``mirror_padding`` (a ``(z, y, x)`` tuple) on-the-fly to its spatial axes.

    Returns a new list; does not mutate ``raws``.
    """
    padded_raws = []
    for raw in raws:
        if raw.ndim == 4:
            pad_width = ((0, 0),) + tuple((int(p), int(p)) for p in mirror_padding)
        elif raw.ndim == 3:
            pad_width = tuple((int(p), int(p)) for p in mirror_padding)
        else:
            raise AssertionError(
                f'Raw dataset must be 3D (DxHxW) or 4D (CxDxHxW); got ndim={raw.ndim}'
            )
        padded_raws.append(_ReflectPaddedView(raw, pad_width))
    return padded_raws


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


def calculate_stats(images):
    """
    Calculates min, max, mean, std given a list of ndarrays
    """
    # flatten first since the images might not be the same size
    flat = np.concatenate(
        [img.ravel() for img in images]
    )
    return np.min(flat), np.max(flat), np.mean(flat), np.std(flat)


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
