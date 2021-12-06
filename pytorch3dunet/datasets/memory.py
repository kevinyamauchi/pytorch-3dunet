import glob
import os
from itertools import chain
from multiprocessing import Lock

import numpy as np

import pytorch3dunet.augment.transforms as transforms
from pytorch3dunet.datasets.utils import get_slice_builder, ConfigDataset, calculate_stats, sample_instances
from pytorch3dunet.unet3d.utils import get_logger

logger = get_logger('MemoryDataset')
lock = Lock()


class MemoryDataset(ConfigDataset):
    """
    Implementation of torch.utils.data.Dataset backed by an input Numpy Array, which iterates over raw datasets
    patch by patch with a given stride.
    This class is only for prediction. NOT to train, validate or test the algorithm.
    """

    def __init__(self, raws, phase,
                 slice_builder_config,
                 transformer_config,
                 mirror_padding=(16, 32, 32),
                 instance_ratio=None,
                 random_seed=0):
        """
        :param raws: list of np.arrays containing raw data
        :param phase: Should only be 'test' for testing TODO: Verify if needed
        :param slice_builder_config: configuration of the SliceBuilder
        :param transformer_config: data augmentation configuration
        :param mirror_padding (int or tuple): number of voxels padded to each axis
        :param instance_ratio (float): A number between (0, 1]: specifies a fraction of ground truth instances to be
            sampled from the dense ground truth labels. TODO: Verify if needed
        :param random_seed: Random seed TODO: Verify if needed
        """
        if mirror_padding is not None:
            if isinstance(mirror_padding, int):
                mirror_padding = (mirror_padding,) * 3
            else:
                assert len(mirror_padding) == 3, f"Invalid mirror_padding: {mirror_padding}"
        # File_path defined for simplicity. Its printed sometimes to explain where the data is extracted from.
        self.file_path = "memory"
        self.mirror_padding = mirror_padding
        self.phase = phase

        self.instance_ratio = instance_ratio

        self.raws = raws

        min_value, max_value, mean, std = self.ds_stats()

        self.transformer = transforms.get_transformer(transformer_config, min_value=min_value, max_value=max_value,
                                                      mean=mean, std=std)
        self.raw_transform = self.transformer.raw_transform()

        # Only for predictions so ignore the label dataset
        self.labels = None
        self.weight_maps = None

        # add mirror padding if needed
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

        # build slice indices for raw and label data sets
        slice_builder = get_slice_builder(self.raws, self.labels, self.weight_maps, slice_builder_config)
        self.raw_slices = slice_builder.raw_slices
        self.label_slices = slice_builder.label_slices
        self.weight_slices = slice_builder.weight_slices

        self.patch_count = len(self.raw_slices)
        logger.info(f'Number of patches: {self.patch_count}')

    def ds_stats(self):
        # calculate global min, max, mean and std for normalization
        min_value, max_value, mean, std = calculate_stats(self.raws)
        logger.info(f'Input stats: min={min_value}, max={max_value}, mean={mean}, std={std}')
        return min_value, max_value, mean, std

    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration

        # get the slice for a given index 'idx'
        raw_idx = self.raw_slices[idx]
        # get the raw data patch for a given slice
        raw_patch_transformed = self._transform_patches(self.raws, raw_idx, self.raw_transform)

        # discard the channel dimension in the slices: predictor requires only the spatial dimensions of the volume
        if len(raw_idx) == 4:
            raw_idx = raw_idx[1:]
        return raw_patch_transformed, raw_idx

    @staticmethod
    def _transform_patches(datasets, raw_idx, transformer):
        transformed_patches = []
        for dataset in datasets:
            # get the raw index slice data and apply the transformer
            transformed_patch = transformer(dataset[raw_idx])
            transformed_patches.append(transformed_patch)

        # if transformed_patches is a singleton list return the first element only
        if len(transformed_patches) == 1:
            return transformed_patches[0]
        else:
            return transformed_patches

    def __len__(self):
        return self.patch_count

    @classmethod
    def create_datasets(cls, dataset_config, phase, raw_dataset):
        phase_config = dataset_config[phase]

        # load data augmentation configuration
        transformer_config = phase_config['transformer']
        # load slice builder config
        slice_builder_config = phase_config['slice_builder']

        # load instance sampling configuration
        instance_ratio = phase_config.get('instance_ratio', None)
        random_seed = phase_config.get('random_seed', 0)

        try:
            logger.info(f'Loading {phase} set from in-memory dataset.')
            dataset = cls(raws=raw_dataset,
                          phase=phase,
                          slice_builder_config=slice_builder_config,
                          transformer_config=transformer_config,
                          mirror_padding=dataset_config.get('mirror_padding', None),
                          instance_ratio=instance_ratio, random_seed=random_seed)
        except Exception:
            logger.error(f'Skipping {phase} set from in-memory dataset.', exc_info=True)
        return [dataset]

    @staticmethod
    def traverse_h5_paths(file_paths):
        assert isinstance(file_paths, list)
        results = []
        for file_path in file_paths:
            if os.path.isdir(file_path):
                # if file path is a directory take all H5 files in that directory
                iters = [glob.glob(os.path.join(file_path, ext)) for ext in ['*.h5', '*.hdf', '*.hdf5', '*.hd5']]
                for fp in chain(*iters):
                    results.append(fp)
            else:
                results.append(file_path)
        return results
