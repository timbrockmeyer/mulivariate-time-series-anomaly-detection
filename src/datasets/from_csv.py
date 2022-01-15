import os
import numpy as np
import torch

from ..data import SlidingWindowDataset

class From_csv(SlidingWindowDataset):

    '''
    Requires CSV files for training and testing. 
    Test file is expected to have its labels in the last column, train file to be without labels.
    Naming:
        Training data: 'train.csv'
        Test data: 'test.csv'
    '''

    def __init__(self, window_size=1, stride=1, horizon=1, train=True, transform=None, target_transform=None, normalize=False, device='cpu'):
         
        self.device = device

        self.name = 'from_csv'

        train_file = 'train.csv'
        test_file = 'test.csv'

        self.meta = (None, None)
        self.normalize = False

        root = os.path.dirname(__file__)       
        raw_dir = os.path.join(root, f'files/raw/{self.name}')
        raw_paths = [os.path.join(raw_dir, ending) for ending in [train_file, test_file]]
        
        if train:
            data = np.genfromtxt(raw_paths[0], delimiter=",")[1:,1:]
            data = torch.from_numpy(data).float().to(self.device)
            labels = None

        else:
            data = np.genfromtxt(raw_paths[1], delimiter=",")[1:,1:]
            data = torch.from_numpy(data).float().to(self.device)
            data, labels = data[:,:-1], data[:,-1]

        super().__init__(data, window_size, stride=stride, horizon=horizon, labels=labels, transform=transform, target_transform=target_transform, normalize=False, device=device)