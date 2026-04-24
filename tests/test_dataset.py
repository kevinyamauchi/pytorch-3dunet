import os
from tempfile import NamedTemporaryFile

import h5py
import numpy as np
from torch.utils.data import DataLoader
import pytest
from pytorch3dunet.datasets.hdf5 import StandardHDF5Dataset, AbstractHDF5Dataset
from pytorch3dunet.datasets.memory import MemoryDataset
from pytorch3dunet.datasets.utils import _ReflectPaddedView, apply_lazy_mirror_padding


class TestReflectPaddedView:
    """
    ``_ReflectPaddedView`` exposes a lazy, reflect-padded view of an underlying array-like.
    It must be bit-for-bit equivalent to ``np.pad(inner, ..., mode='reflect')[slice]`` for any
    patch slice the SliceBuilder can produce, but without materialising the full padded volume.
    """

    def test_matches_numpy_pad_3d(self):
        np.random.seed(0)
        # Interior patches, boundary patches on either side, and straddling patches, across a
        # range of shapes / pad sizes. 2500 randomised patches in total.
        configs = [((50, 80, 90), (16, 32, 32)),
                   ((32, 64, 64), (16, 32, 32)),
                   ((40, 100, 120), (10, 20, 20)),
                   ((33, 65, 67), (16, 32, 32)),
                   ((80, 80, 80), (32, 32, 32))]
        for shape, pad in configs:
            vol = np.random.randn(*shape).astype('float32')
            pad_width = tuple((p, p) for p in pad)
            ref = np.pad(vol, pad_width, mode='reflect')
            view = _ReflectPaddedView(vol, pad_width)
            assert view.shape == ref.shape
            assert view.ndim == ref.ndim
            assert view.dtype == ref.dtype
            for _ in range(500):
                sizes = [np.random.randint(max(1, pad[a]), 3 * pad[a] + 1) for a in range(3)]
                sizes = [min(s, ref.shape[a]) for a, s in enumerate(sizes)]
                starts = [np.random.randint(0, ref.shape[a] - sizes[a] + 1) for a in range(3)]
                slc = tuple(slice(starts[a], starts[a] + sizes[a]) for a in range(3))
                np.testing.assert_array_equal(view[slc], ref[slc])

    def test_matches_numpy_pad_4d_channels(self):
        np.random.seed(1)
        vol = np.random.randn(3, 40, 80, 80).astype('float32')
        pad_width = ((0, 0), (16, 16), (32, 32), (32, 32))
        ref = np.pad(vol, pad_width, mode='reflect')
        view = _ReflectPaddedView(vol, pad_width)
        assert view.shape == ref.shape
        for _ in range(300):
            sizes = [3,
                     np.random.randint(16, 50),
                     np.random.randint(32, 100),
                     np.random.randint(32, 100)]
            starts = [0] + [np.random.randint(0, ref.shape[a] - sizes[a] + 1) for a in range(1, 4)]
            slc = tuple(slice(starts[a], starts[a] + sizes[a]) for a in range(4))
            np.testing.assert_array_equal(view[slc], ref[slc])

    def test_apply_lazy_mirror_padding_wraps_shapes(self):
        raws_3d = [np.zeros((40, 80, 80), dtype='float32')]
        wrapped = apply_lazy_mirror_padding(raws_3d, (16, 32, 32))
        assert wrapped[0].shape == (72, 144, 144)
        assert wrapped[0].ndim == 3

        raws_4d = [np.zeros((2, 40, 80, 80), dtype='float32')]
        wrapped = apply_lazy_mirror_padding(raws_4d, (16, 32, 32))
        assert wrapped[0].shape == (2, 72, 144, 144)
        assert wrapped[0].ndim == 4

    def test_does_not_materialise_full_volume(self):
        """
        Sanity-check that the view only reads the sub-region it needs on ``__getitem__``
        rather than reading the whole backing array. We track access via a counter wrapper.
        """

        class CountingArray:
            def __init__(self, arr):
                self._arr = arr
                self.shape = arr.shape
                self.ndim = arr.ndim
                self.dtype = arr.dtype
                self.last_key = None

            def __getitem__(self, key):
                self.last_key = key
                return self._arr[key]

        inner = np.random.rand(100, 100, 100).astype('float32')
        counter = CountingArray(inner)
        view = _ReflectPaddedView(counter, ((16, 16), (32, 32), (32, 32)))

        # Interior patch: fetch exactly matches the requested slice size (no over-fetch).
        _ = view[slice(50, 66), slice(50, 82), slice(50, 82)]
        fetched_shape = tuple(s.stop - s.start for s in counter.last_key)
        assert fetched_shape == (16, 32, 32), f'interior over-fetch: {fetched_shape}'

        # Pure-boundary patch on one axis: fetch is bounded, not full-volume.
        _ = view[slice(0, 16), slice(40, 72), slice(40, 72)]
        fetched_shape = tuple(s.stop - s.start for s in counter.last_key)
        # axis 0: must include index 0 and enough data for reflect (pad=16 -> slab>=17)
        assert fetched_shape[0] <= 100 and fetched_shape[0] >= 17
        assert fetched_shape[1] == 32
        assert fetched_shape[2] == 32


class TestMemoryDataset:
    """
    Test converting an in-memory numpy array to a pytorch dataset.
    """
    def test_mermory_dataset(self, transformer_config):
        """
        Test random input numpy arrays of floats [0,1) of a given shape.
        Currently tests one input.
        Args:
            transformer_config:

        Returns:

        """
        raws=[]
        for i in range(1):
            raw = np.random.rand(128, 128, 128)
            raws.append(raw)

        patch_shapes = [(127, 127, 127), (69, 70, 70), (32, 64, 64)]
        stride_shapes = [(1, 1, 1), (17, 23, 23), (32, 64, 64)]

        phase = 'test'

        for patch_shape, stride_shape in zip(patch_shapes, stride_shapes):
            dataset = MemoryDataset(raws=raws, phase=phase,
                                    slice_builder_config=_slice_builder_conf(patch_shape, stride_shape),
                                    transformer_config=transformer_config[phase]['transformer'],
                                    mirror_padding=None)

            # create zero-arrays of the same shape as the original dataset in order to verify if every element
            # was visited during the iteration
            visit_raw = np.zeros_like(raws)

            for (_, idx) in dataset:
                visit_raw[idx] = 1

            # verify that every element was visited at least once
            assert np.all(visit_raw)


class TestHDF5Dataset:
    def test_hdf5_dataset(self, transformer_config):
        path = create_random_hdf5_dataset((128, 128, 128))

        patch_shapes = [(127, 127, 127), (69, 70, 70), (32, 64, 64)]
        stride_shapes = [(1, 1, 1), (17, 23, 23), (32, 64, 64)]

        phase = 'test'

        for patch_shape, stride_shape in zip(patch_shapes, stride_shapes):
            with h5py.File(path, 'r') as f:
                raw = f['raw'][...]
                label = f['label'][...]

                dataset = StandardHDF5Dataset(path, phase=phase,
                                              slice_builder_config=_slice_builder_conf(patch_shape, stride_shape),
                                              transformer_config=transformer_config[phase]['transformer'],
                                              mirror_padding=None,
                                              raw_internal_path='raw',
                                              label_internal_path='label')

                # create zero-arrays of the same shape as the original dataset in order to verify if every element
                # was visited during the iteration
                visit_raw = np.zeros_like(raw)
                visit_label = np.zeros_like(label)

                for (_, idx) in dataset:
                    visit_raw[idx] = 1
                    visit_label[idx] = 1

                # verify that every element was visited at least once
                assert np.all(visit_raw)
                assert np.all(visit_label)

    def test_hdf5_with_multiple_label_datasets(self, transformer_config):
        path = create_random_hdf5_dataset((128, 128, 128), label_datasets=['label1', 'label2'])
        patch_shape = (32, 64, 64)
        stride_shape = (32, 64, 64)
        phase = 'train'
        dataset = StandardHDF5Dataset(path, phase=phase,
                                      slice_builder_config=_slice_builder_conf(patch_shape, stride_shape),
                                      transformer_config=transformer_config[phase]['transformer'],
                                      raw_internal_path='raw',
                                      label_internal_path=['label1', 'label2'])

        for raw, labels in dataset:
            assert len(labels) == 2

    def test_hdf5_with_multiple_raw_and_label_datasets(self, transformer_config):
        path = create_random_hdf5_dataset((128, 128, 128), raw_datasets=['raw1', 'raw2'],
                                          label_datasets=['label1', 'label2'])
        patch_shape = (32, 64, 64)
        stride_shape = (32, 64, 64)
        phase = 'train'
        dataset = StandardHDF5Dataset(path, phase=phase,
                                      slice_builder_config=_slice_builder_conf(patch_shape, stride_shape),
                                      transformer_config=transformer_config[phase]['transformer'],
                                      raw_internal_path=['raw1', 'raw2'], label_internal_path=['label1', 'label2'])

        for raws, labels in dataset:
            assert len(raws) == 2
            assert len(labels) == 2

    def test_augmentation(self, transformer_config):
        raw = np.random.rand(32, 96, 96)
        # assign raw to label's channels for ease of comparison
        label = np.stack(raw for _ in range(3))
        # create temporary h5 file
        tmp_file = NamedTemporaryFile()
        tmp_path = tmp_file.name
        with h5py.File(tmp_path, 'w') as f:
            f.create_dataset('raw', data=raw)
            f.create_dataset('label', data=label)

        # set phase='train' in order to execute the train transformers
        phase = 'train'
        dataset = StandardHDF5Dataset(tmp_path, phase=phase,
                                      slice_builder_config=_slice_builder_conf((16, 64, 64), (8, 32, 32)),
                                      transformer_config=transformer_config[phase]['transformer'])

        # test augmentations using DataLoader with 4 worker threads
        data_loader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=True)
        for (img, label) in data_loader:
            for i in range(label.shape[0]):
                assert np.allclose(img, label[i])

    def test_traverse_file_paths(self, tmpdir):
        test_tmp_dir = os.path.join(tmpdir, 'test')
        os.mkdir(test_tmp_dir)

        expected_files = [
            os.path.join(tmpdir, 'f1.h5'),
            os.path.join(test_tmp_dir, 'f2.h5'),
            os.path.join(test_tmp_dir, 'f3.hdf'),
            os.path.join(test_tmp_dir, 'f4.hdf5'),
            os.path.join(test_tmp_dir, 'f5.hd5')
        ]
        # create expected files
        for ef in expected_files:
            with h5py.File(ef, 'w') as f:
                f.create_dataset('raw', data=np.random.randn(4, 4, 4))

        # make sure that traverse_file_paths runs correctly
        file_paths = [os.path.join(tmpdir, 'f1.h5'), test_tmp_dir]
        actual_files = AbstractHDF5Dataset.traverse_h5_paths(file_paths)

        assert expected_files == actual_files


def create_random_hdf5_dataset(shape, ignore_index=False, raw_datasets=None, label_datasets=None):
    if label_datasets is None:
        label_datasets = ['label']
    if raw_datasets is None:
        raw_datasets = ['raw']

    tmp_file = NamedTemporaryFile(delete=False)

    with h5py.File(tmp_file.name, 'w') as f:
        for raw_dataset in raw_datasets:
            f.create_dataset(raw_dataset, data=np.random.rand(*shape))

        for label_dataset in label_datasets:
            if ignore_index:
                f.create_dataset(label_dataset, data=np.random.randint(-1, 2, shape))
            else:
                f.create_dataset(label_dataset, data=np.random.randint(0, 2, shape))

    return tmp_file.name


def _slice_builder_conf(patch_shape, stride_shape):
    return {
        'name': 'SliceBuilder',
        'patch_shape': patch_shape,
        'stride_shape': stride_shape
    }
