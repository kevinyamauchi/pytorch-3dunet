import argparse

import torch
import yaml

from pytorch3dunet.unet3d import utils

logger = utils.get_logger('ConfigLoader')


def load_config():
    """
    Parse arguments and load device from string
    Returns:
        config dictionary loaded form parsed file location
    """
    parser = argparse.ArgumentParser(description='UNet3D')
    parser.add_argument('--config', type=str, help='Path to the YAML config file', required=True)
    args = parser.parse_args()
    config = _load_config_yaml(args.config)

    config['device'] = get_device(config.get('device', None))
    return config


def get_device(device_str):
    """
    Load device to train or test on. If no device is defined, then automatically load the available device (pref. GPU)
    Args:
        device_str: string
            name of device to load using torch

    Returns:
        torch device object
    """
    # Get a device to train on
    if device_str is not None:
        logger.info(f"Device specified in config: '{device_str}'")
        if device_str.startswith('cuda') and not torch.cuda.is_available():
            logger.warn('CUDA not available, using CPU')
            device_str = 'cpu'
    else:
        device_str = "cuda:0" if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using '{device_str}' device")

    return torch.device(device_str)


def _load_config_yaml(config_file):
    return yaml.safe_load(open(config_file, 'r'))
