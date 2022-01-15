import os
import json
import numpy as np
import pandas as pd
import torch
from shutil import rmtree

from ..data import SlidingWindowDataset


class Swat(SlidingWindowDataset):

    '''
    LOAD ORIGINAL FILES IN EXCEL FIRST!!!
    delete the unnecessary first rows and save as a CSV file.

    Dataset can be requested from 
    https://itrust.sutd.edu.sg/testbeds/secure-water-treatment-swat/
    '''

    def __init__(self, window_size=1, stride=1, horizon=1, train=True, transform=None, target_transform=None, normalize=False, device='cpu'):
         
        self.device = device

        self.name = 'swat'

        train_file = 'SWaT_Dataset_Normal_v0.csv'
        test_file = 'SWaT_Dataset_Attack_v0.csv'

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
            raise Exception(f'{self.name} dataset raw file processing failed')

        return data, labels, sensor_list

    def process(self):
        
        # purge old files if any exist
        if os.path.exists(self.processed_dir):
            rmtree(self.processed_dir)
        
        files = {'train': self.raw_paths[0], 'test': self.raw_paths[1]}

        for key, file in files.items():
                    
            df = pd.read_csv(file)

            # strip white spaces from column names
            df = df.rename(columns=lambda x: x.strip())

            # timestamp column to index
            df.iloc[:,0] = df.index
            df = df.set_index(df.columns[0])

            if key == 'train':
                # drop label column for training data
                df = df.drop(df.columns[-1], axis=1)

                column_names = df.columns.to_numpy()

                train_data = df.to_numpy()
                train_data = torch.from_numpy(train_data).float().to(self.device)

            else:
                # categorial labels to numerical values
                vocab = {'Normal': 0, 'Attack': 1, 'A ttack': 1}
                df.iloc[:,-1] = df.iloc[:,-1].apply(lambda x: vocab[x])

                test_data = df.to_numpy()
                test_data = torch.from_numpy(test_data).float().to(self.device)
                test_data, test_labels = test_data[:,:-1], test_data[:,-1]

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

        os.makedirs(self.processed_dir, exist_ok=True)
        torch.save(train_data, self.processed_paths[0])
        torch.save(test_data, self.processed_paths[1])
        torch.save(test_labels, self.processed_paths[2])
        np.savetxt(self.processed_paths[3], column_names, fmt = "%s")
        dump = json.dumps(meta, indent=4)
        with open(self.processed_paths[4], 'w') as f:
            f.write(dump) 
