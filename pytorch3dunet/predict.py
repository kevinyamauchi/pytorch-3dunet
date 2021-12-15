import importlib
import os

import torch
import torch.nn as nn

from pytorch3dunet.datasets.utils import get_test_loaders
from pytorch3dunet.unet3d import utils
from pytorch3dunet.unet3d.config import load_config
from pytorch3dunet.unet3d.model import get_model

logger = utils.get_logger('UNet3DPredict')


def _get_predictor(model, output_dir, config):
    predictor_config = config.get('predictor', {})
    class_name = predictor_config.get('name', 'StandardPredictor')

    m = importlib.import_module('pytorch3dunet.unet3d.predictor')
    predictor_class = getattr(m, class_name)

    return predictor_class(model, output_dir, config, **predictor_config)


def run_predictions(config, raw_dataset=None):
    """
    Run prediction pipeline using config dictionary and if a in-memory dataset is passed on then return that one.
    Args:
        config: Dict - Configuration dictionary (includes input file location if no dataset type is not MemoryDataset)
        raw_dataset: List[np.array] Raw dataset which is already loaded in memory

    Returns:
        results_list: List(Dict("name":data)) - Returns the list of predictions as a dictionary
    """
    # Create the model
    model = get_model(config['model'])

    # Load model state
    model_path = config['model_path']
    if model_path == "":
        logger.info('No model to load from path')
    else:
        logger.info(f'Loading model from {model_path}...')
        utils.load_checkpoint(model_path, model)

    # use DataParallel if more than 1 GPU available
    device = config['device']
    if torch.cuda.device_count() > 1 and not device.type == 'cpu':
        model = nn.DataParallel(model)
        logger.info(f'Using {torch.cuda.device_count()} GPUs for prediction')

    logger.info(f"Sending the model to '{device}'")
    model = model.to(device)

    output_dir = config['loaders'].get('output_dir', None)
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f'Saving predictions to: {output_dir}')

    # create predictor instance
    predictor = _get_predictor(model, output_dir, config)

    result_list = []
    for test_loader in get_test_loaders(config, raw_dataset):
        # run the model prediction on the test_loader and save the results in the output_dir
        results = predictor(test_loader)
        # If an in-memory dataset is passed then also save results in memory to be able to return them in-memory
        result_list.append(results)

    # Only return results if in-memory dataset is passed
    if raw_dataset:
        logger.info(f'length of results list: {len(result_list)}, dict keys of first list entry {result_list[0].keys()}')
        return result_list
    else:
        return None


if __name__ == '__main__':
    # Load configuration
    config = load_config()

    run_predictions(config)
