import torch
from torch.utils.data import Dataset
import numpy as np

class DummyDataset(Dataset):
    def __init__(self, shape, label):
        self.data = torch.FloatTensor(np.random.normal(0,1,shape))
        self.label = torch.FloatTensor(np.random.normal(0,1,28*28))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, item):
        return self.data[item], self.label[item]
