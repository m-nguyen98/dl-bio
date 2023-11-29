from abc import ABC

import numpy as np
from torch.utils.data import DataLoader

from datasets.dataset import *

class LCDataset(FewShotDataset, ABC):
    _dataset_name = 'livecell'
    _dataset_url = 'http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/images.zip'
    
    def load_tabular_muris(self, mode='train', min_samples=20):
        train_cell_types = []
        val_cell_types = []
        test_cell_types = []
        split = {'train': train_cell_types,
                 'val': val_cell_types,
                 'test': test_cell_types}
    
        cell_types = split[mode]
        
        # subset data based on target cell type
        adata = adata[adata.obs['tissue'].isin(tissues)]

        filtered_index = adata.obs.groupby(["label"]) \
            .filter(lambda group: len(group) >= min_samples) \
            .reset_index()['index']
        adata = adata[filtered_index]

        # convert gene to torch tensor x
        samples = adata.to_df().to_numpy(dtype=np.float32)
        # convert label to torch tensor y
        targets = adata.obs['label'].cat.codes.to_numpy(dtype=np.int32)
        # go2gene = get_go2gene(adata=adata, GO_min_genes=32, GO_max_genes=None, GO_min_level=6, GO_max_level=1)
        # go_mask = create_go_mask(adata, go2gene)
        return samples, targets