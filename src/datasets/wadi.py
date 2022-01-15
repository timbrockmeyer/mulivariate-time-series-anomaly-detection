import os
import json
import numpy as np
import pandas as pd
import torch
from shutil import rmtree

from ..data import SlidingWindowDataset


class Wadi(SlidingWindowDataset):
    
    ''' LOAD ORIGINAL FILES IN EXCEL FIRST!!!
    delete the unnecessary first rows and save as a CSV file.

    The dataset includes a PDF with descriptions of 15 anomaly events,
    including start and end dates (m/d/y) and times.
    -> copy the following table and save as "WADI_attacktimes.csv":

    Start_Date	Start_Time	End_Date	End_Time
    10/9/2017	19:25:00	10/9/2017	19:50:16
    10/10/2017	10:24:10	10/10/2017	10:34:00
    10/10/2017	10:55:00	10/10/2017	11:24:00
    10/10/2017	11:30:40	10/10/2017	11:44:50
    10/10/2017	13:39:30	10/10/2017	13:50:40
    10/10/2017	14:48:17	10/10/2017	14:59:55
    10/10/2017	17:40:00	10/10/2017	17:49:40
    10/10/2017	10:55:00	10/10/2017	10:56:27
    10/11/2017	11:17:54	10/11/2017	11:31:20
    10/11/2017	11:36:31	10/11/2017	11:47:00
    10/11/2017	11:59:00	10/11/2017	12:05:00
    10/11/2017	12:07:30	10/11/2017	12:10:52
    10/11/2017	12:16:00	10/11/2017	12:25:36
    10/11/2017	15:26:30	10/11/2017	15:37:00

    Data can be requested from
    https://itrust.sutd.edu.sg/itrust-labs-home/itrust-labs_wadi/
    '''

    def __init__(self, window_size=1, stride=1, horizon=1, train=True, transform=None, target_transform=None, normalize=False, device='cpu'):
         
        self.device = device

        self.name = 'wadi'

        train_file = 'WADI_14days.csv'
        test_file = 'WADI_attackdata.csv'
        label_file = 'WADI_attacktimes.csv'

        root = os.path.dirname(__file__)       
        raw_dir = os.path.join(root, f'files/raw/{self.name}')
        self.processed_dir = os.path.join(root, f'files/processed/{self.name}')

        self.raw_paths = [os.path.join(raw_dir, ending) for ending in [train_file, test_file, label_file]]
        self.processed_paths = [os.path.join(self.processed_dir, ending) for ending in ['train.pt', 'test.pt', 'labels.pt', 'list.txt', 'meta.json']]
        
        data, labels, sensor_names = self.load(train)

        self.node_names = sensor_names

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

        df_train = pd.read_csv(self.raw_paths[0]) 
        df_test = pd.read_csv(self.raw_paths[1])
        anomaly_timeframes = pd.read_csv(self.raw_paths[2])

        def drop_columns(columns):
            df_train.drop(columns, axis=1, inplace=True)
            df_test.drop(columns, axis=1, inplace=True)

        assert list(df_train.columns) == list(df_test.columns)

        # find row indices of anomaly interval
        start_indices = pd.merge(df_test, anomaly_timeframes, left_on=['Date', 'Time'], right_on=['Start_Date', 'Start_Time'])['Row']
        end_indices = pd.merge(df_test, anomaly_timeframes, left_on=['Date', 'Time'], right_on=['End_Date', 'End_Time'])['Row']
        assert start_indices.shape == end_indices.shape

        # add anomaly labels to test data  
        labels = pd.Series(np.zeros(len(df_test)))
        for a,b in zip(start_indices,end_indices):
            labels[a:b+1] = np.ones((b-a)+1)
        df_test['label'] = labels

        # drop date and time columns
        datetime_cols = ['Date', 'Time']
        drop_columns(datetime_cols)

        # fix columns
        for df in [df_train, df_test]:        
            # set index column
            df.rename(columns={df.columns[0]:'timestamp'}, inplace=True)
            df.iloc[:,0] = df.index
            df.set_index(df.columns[0], inplace=True)
            # strip column names
            df.rename(columns=lambda x: x.strip(), inplace=True)
            # shorten column names
            df.columns = [x.split('\\')[-1] for x in df.columns]

        # account for missing data
        # completely empty colums in training or test data
        empty_columns = [col for col in df_train.columns if df_train[col].isnull().all() or df_test[col].isnull().all()]
        drop_columns(empty_columns)
        # other missing values
        assert not df_test.isnull().any().any()
        df_train = df_train.interpolate(method='nearest')

        # columns with zero variance in test data
        zero_var_columns_test = [col for col in df_test.columns if df_test[col].var() == 0]
        # columns with extremly high variance in training data
        extreme_var_columns_train = [col for col in df_train.columns if df_train[col].var() > 10000]
        drop_columns(zero_var_columns_test + extreme_var_columns_train)

        assert list(df_train.columns) == list(df_test.columns)[:-1]

        column_names = df_train.columns.to_numpy()

        train_data = df_train.to_numpy()
        train_data = torch.from_numpy(train_data).float().to(self.device)

        test_data = df_test.to_numpy()
        test_data = torch.from_numpy(test_data).float().to(self.device)

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

        test_data, test_labels = test_data[:,:-1], test_data[:,-1]

        assert train_data.size(1) == test_data.size(1)

        os.makedirs(self.processed_dir, exist_ok=True)
        torch.save(train_data, self.processed_paths[0])
        torch.save(test_data, self.processed_paths[1])
        torch.save(test_labels, self.processed_paths[2])
        np.savetxt(self.processed_paths[3], column_names, fmt = "%s")
        dump = json.dumps(meta, indent=4)
        with open(self.processed_paths[4], 'w') as f:
            f.write(dump) 














