import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from datasets.dataset import *

class LCDataset(FewShotDataset):
    _dataset_name = 'livecell'
    _dataset_url = 'http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/images.zip'
    transform = T.Compose([
        T.ToPILImage(),
        T.RandomResizedCrop((224,224)),
        T.ToTensor()])
    x_dim = 366080
    
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
            file_label = filename.split('_')[0]
            if filename.endswith('tif') and file_label in cell_types:
                file_names.append(filename)
                labels.append(self.mapping.index(file_label))
    
        return file_names, labels
    
    
class LCSimpleDataset(LCDataset):
    def __init__(self, batch_size, root='./data/', mode='train', min_samples=20):
        self.initialize_data_dir(root, download_flag=False)
        self.samples, self.targets = self.load_livecell(mode, min_samples)
        self.batch_size = batch_size 
        self.x_dim = 366080
        
        super().__init__()
        
    def __getitem__(self, i):
        filename = os.path.join(self._data_dir, self.samples[i])
        img = Image.open(filename)
        tensor_input = TF.to_tensor(img)
        X = torch.squeeze(self.transform(tensor_input))
        X = X.unsqueeze(0)
        
        return X, self.targets[i]
    
    
    def __len__(self):
        return len(self.samples)
    
    @property
    def dim(self):
        return self.x_dim
    
    def get_data_loader(self) -> DataLoader:
        data_loader_params = dict(batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        data_loader = torch.utils.data.DataLoader(self, **data_loader_params)

        return data_loader

class LCSetDataset(LCDataset):
    def __init__(self, n_way, n_support, n_query, n_episode=100, root='./data', mode='train'):
        self.initialize_data_dir(root, download_flag=False)
        
        self.n_way = n_way
        self.n_episode = n_episode
        min_samples = n_support + n_query
        
        self.file_names, self.labels = self.load_livecell(mode, min_samples)
        self.categories = np.unique(self.labels)  # Unique cell labels
        
        self.sub_dataloader = []

        sub_data_loader_params = dict(batch_size=min_samples,
                                      shuffle=True,
                                      num_workers=0,  # use main thread only or may receive multiple batches
                                      pin_memory=False)

        for cl in self.categories:
            cl_list = []
            for filename in self.file_names:
                file_label = filename.split('_')[0]
                if file_label == cl:
                    cl_list.append(filename)
                    
            sub_dataset = FewShotSubDataset(np.array(cl_list), self.mapping.index(cl))
            self.sub_dataloader.append(torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params))

        super().__init__()
        
    def __getitem__(self, i):
        batch_filenames, batch_labels = next(iter(self.sub_dataloader[i]))
        
        x_list = []
        for name, label in zip(batch_filenames, batch_labels):
            filename = os.path.join(self._data_dir, name)
            img = Image.open(filename)
            tensor_input = TF.to_tensor(img)
            X = torch.squeeze(self.transform(tensor_input))
            X = X.unsqueeze(0)
            x_list.append(X)
           
        return x_list, batch_labels
    
    def __len__(self):
        return len(self.categories)
    
    @property
    def dim(self):
        return self.x_dim
    
    def get_data_loader(self) -> DataLoader:
        sampler = EpisodicBatchSampler(len(self), self.n_way, self.n_episode)
        data_loader_params = dict(batch_sampler=sampler, num_workers=4, pin_memory=True)
        data_loader = torch.utils.data.DataLoader(self, **data_loader_params)
        return data_loader
    