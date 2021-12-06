import os
from tempfile import NamedTemporaryFile

import h5py
import numpy as np
import torch
import pytest

from pytorch3dunet.predict import run_predictions
from pytorch3dunet.unet3d.utils import remove_halo
from pytorch3dunet.unet3d.config import get_device


class TestPredictor:
    def test_standard_predictor(self, tmpdir, test_config):
        # Add output dir
        test_config['loaders']['output_dir'] = tmpdir

        # create random dataset
        tmp = NamedTemporaryFile(delete=False)

        with h5py.File(tmp.name, 'w') as f:
            shape = (32, 64, 64)
            f.create_dataset('raw', data=np.random.rand(*shape))

        # Add input file
        test_config['loaders']['test']['file_paths'] = [tmp.name]

        # Initialize device
        test_config['device'] = get_device(test_config.get('device', None))

        run_predictions(test_config)

        assert os.path.exists(os.path.join(tmpdir, os.path.split(tmp.name)[1] + '_predictions.h5'))

    def test_standard_predictor_memory_dataset(self, tmpdir, test_config):
        # Add output dir
        test_config['loaders']['output_dir'] = tmpdir
        # Initialize device
        test_config['device'] = get_device(test_config.get('device', None))
        # Set dataset Type to MemoryDataset
        test_config['loaders']['dataset'] = "MemoryDataset"

        raw_dataset = []
        shape = (32, 64, 64)
        for image in range(1):
            raw_dataset.append(np.random.rand(*shape))

        result_list = run_predictions(test_config, raw_dataset=raw_dataset)

        assert result_list is not None
        assert os.path.exists(os.path.join(tmpdir, 'memory_predictions.h5'))

    def test_remove_halo(self):
        patch_halo = (4, 4, 4)
        shape = (128, 128, 128)
        input = np.random.randint(0, 10, size=(1, 16, 16, 16))

        index = (slice(0, 1), slice(12, 28), slice(16, 32), slice(16, 32))
        u_patch, u_index = remove_halo(input, index, shape, patch_halo)

        assert np.array_equal(input[:, 4:12, 4:12, 4:12], u_patch)
        assert u_index == (slice(0, 1), slice(16, 24), slice(20, 28), slice(20, 28))

        index = (slice(0, 1), slice(112, 128), slice(112, 128), slice(112, 128))
        u_patch, u_index = remove_halo(input, index, shape, patch_halo)

        assert np.array_equal(input[:, 4:16, 4:16, 4:16], u_patch)
        assert u_index == (slice(0, 1), slice(116, 128), slice(116, 128), slice(116, 128))
