from datasets.modelnet import get_dataloader_modelnet40
from datasets.data_utils import *


def get_dataloader(split, config):
    if config.dataset_name == 'modelnet40':
        return get_dataloader_modelnet40(split, config)
    else:
        raise ValueError
    