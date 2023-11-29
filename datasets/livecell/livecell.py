import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader

from datasets.dataset import *

class LCDataset(FewShotDataset):
    _dataset_name = 'livecell'
    _dataset_url = 'http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/images.zip'
    
    def __init__(self, batch_size, root='./data/', mode='train', min_samples=20):
        self.initialize_data_dir(root, download_flag=False)
        self.file_names, self.labels = self.load_livecell(mode, min_samples)
        self.batch_size = batch_size
        super().__init__()
        
    def __getitem__(self, i):
        filename = self.file_names[i]
        img = Image.open(filename)
        tensor_input = TF.to_tensor(filename)
        return tensor_input, self.labels[i]
    
    def __len__(self):
        return len(self.file_names)
    
    @property
    def dim(self):
        return 366080
    
    def get_data_loader(self) -> DataLoader:
        data_loader_params = dict(batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        data_loader = torch.utils.data.DataLoader(self, **data_loader_params)

        return data_loader
    
    def load_livecell(self, mode='train', min_samples=20):
        train_cell_types = ["A172", "BT474", "BV2", "Huh7"]
        val_cell_types = ["MCF7", "SHSY5Y"]
        test_cell_types = ["SkBr3", "SKOV3"]
        split = {'train': train_cell_types,
                 'val': val_cell_types,
                 'test': test_cell_types}
    
        cell_types = split[mode]
        
        file_names = []
        labels = []
        for filename in os.listdir(self._data_dir):
            if filename.endswith('tif') :
                file_names.append(filename)
                labels.append(filename.split('_')[0])
        
        return file_names, labels