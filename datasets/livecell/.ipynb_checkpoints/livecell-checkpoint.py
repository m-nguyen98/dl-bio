from abc import ABC

import numpy as np
from torch.utils.data import DataLoader

from datasets.dataset import *

class LCDataset(FewShotDataset, ABC):
    _dataset_name = 'livecell'
    _dataset_url = 'http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/images.zip'
    
    