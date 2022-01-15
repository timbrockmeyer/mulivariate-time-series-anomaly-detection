import os
import json
import numpy as np
import torch
from shutil import rmtree

from ..data import SlidingWindowDataset

class Demo(SlidingWindowDataset):

    '''
    Small excerpt from the MSL dataset used for testing    
    '''

    def __init__(self, window_size=1, stride=1, horizon=1, train=True, transform=None, target_transform=None, normalize=False, device='cpu'):
         
        self.device = device

        self.name = 'demo'

        train_file = 'train.csv'
        test_file = 'test.csv'

        root = os.path.dirname(__file__)       
        raw_dir = os.path.join(root, f'files/raw/{self.name}')
        self.processed_dir = os.path.join(root, f'files/processed/{self.name}')

        self.raw_paths = [os.path.join(raw_dir, ending) for ending in [train_file, test_file]]
        self.processed_paths = [os.path.join(self.processed_dir, ending) for ending in ['train.pt', 'test.pt', 'labels.pt', 'list.txt', 'meta.json']]
        
        data, labels, node_names = self.load(train)

        self.node_names = node_names

        super().__init__(data, window_size, stride=stride, horizon=horizon, labels=labels, transform=transform, target_transform=target_transform, normalize=normalize, device=device)

    def load(self, train):

        # process csv files if not done
        if not all(map(lambda x: os.path.isfile(x), self.processed_paths)):
            self.process()
        
        # check if processed and load
        if all(map(lambda x: os.path.isfile(x), self.processed_paths)):
            if train:
                data = torch.load(self.processed_paths[0], map_location=self.device)
                labels = None
            else:
                data = torch.load(self.processed_paths[1], map_location=self.device)
                labels = torch.load(self.processed_paths[2], map_location=self.device)
            sensor_list = np.loadtxt(self.processed_paths[3], dtype=str)
            with open(self.processed_paths[4], 'r') as f:
                self.meta = json.load(f)
        else:
            raise Exception(f'{self.name} dataset file processing failed')

        return data, labels, sensor_list

    def process(self):

        # purge old files if any exist
            if os.path.exists(self.processed_dir):
                rmtree(self.processed_dir)

            # load csv file
            train_csv = np.genfromtxt(self.raw_paths[0], delimiter=",")
            train_data = torch.from_numpy(train_csv[1:,1:]).float().to(self.device)
            
            test_csv = np.genfromtxt(self.raw_paths[1], delimiter=",")
            test_data = torch.from_numpy(test_csv[1:,1:]).float().to(self.device)
            test_data, test_labels = test_data[:,:-1], test_data[:,-1]

            with open(self.raw_paths[0], 'r') as f:
                line = f.readline().split(',')[1:]
            sensor_list = np.array(list(map(str.strip, line)), dtype=str)

            meta = [self.name, {
                'num_nodes': train_data.size(1),
                'train': {
                    'samples': train_data.size(0),
                    'min': train_data.min(dim=0)[0].tolist(),
                    'max': train_data.max(dim=0)[0].tolist(), 
                },
                'test': {
                    'samples': test_data.size(0),
                    'min': test_data.min(dim=0)[0].tolist(),
                    'max': test_data.max(dim=0)[0].tolist(), 
                }
            }]

            os.makedirs(self.processed_dir)
            torch.save(train_data, self.processed_paths[0])
            torch.save(test_data, self.processed_paths[1])
            torch.save(test_labels, self.processed_paths[2])
            np.savetxt(self.processed_paths[3], sensor_list, delimiter='\n', fmt='%s')
            dump = json.dumps(meta, indent=4)
            with open(self.processed_paths[4], 'w') as f:
                f.write(dump) 
