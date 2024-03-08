import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset, random_split
import random

class GaussianNoisyDataset(Dataset):
    """
    Dataset wrapper to add Gaussian noise to data.
    """
    def __init__(self, dataset, mean=0., std=1.):
        self.dataset = dataset
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, target = self.dataset[idx]
        
        noise = torch.randn_like(data) * self.std + self.mean
        noisy_data = data + noise
        
        return noisy_data, target
    
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False