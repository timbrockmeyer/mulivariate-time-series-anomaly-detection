import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from sklearn.preprocessing import MinMaxScaler

class SlidingWindowDataset(Dataset):
    '''
    Dataset class for multivariate time series data. 
    Each returned sample of the dataset is a sliding window of specific length
    given as a pytorch geometric data objects. 
    https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.Data

    This should serve as the base class for specific datasets
    
    Args:
        data (Tensor): 2d tensor with one timesteps in the rows and sensors as columns.
        window_size (int): Length of the sliding window of each sample.
        stride (int, optional): Stride length a which the dataset is sampled.
        horizon (int, optional): Number of timesteps used as prediction target.
        labels (Tensor, optional): Anomaly labels (None during training).
        transform (callable, optional): Transform appplied to the data.
        target_transform (callable, optional): Transform applied to the labels.
        device (str, optional): Device where data will be held (cpu, cuda).
    '''
    def __init__(self, data, window_size, stride=1, horizon=1, labels=None, transform=None, target_transform=None, normalize=False, device='cpu'):
        
        self.window_size = window_size
        self.stride = stride
        self.horizon = horizon
        self.normalize = normalize
        self.device = torch.device(device)

        self.dataset = self._process(data, labels, transform, target_transform)
                    
    def _process(self, data, labels, transform, target_transform):
        assert isinstance(data, torch.Tensor)
        assert isinstance(labels, (type(None), torch.Tensor))

        _, info = self.meta
        if self.normalize:
            train_meta = info['train']
            min_ = torch.tensor(train_meta['min'], requires_grad=False)
            max_ = torch.tensor(train_meta['max'], requires_grad=False)

            fit_data = torch.stack([min_, max_], dim=0).detach().cpu().numpy()

            normalizer = MinMaxScaler(feature_range=(0,1)).fit(fit_data)

            data = torch.tensor(normalizer.transform(data.cpu().numpy())).to(self.device)

        data = data.to(self.device).T.float()
        
        if transform is not None:
            data = transform(data)

        self.num_nodes = data.size(0)
        
        if labels is not None:
            labels = labels.to(self.device)   
            
            if target_transform is not None:
                labels = target_transform(labels)

        self._len = ((data.size(1) - self.window_size - self.horizon) // self.stride) + 1

        dataset = []
        for idx in range(self._len):
            id = idx
            idx *= self.stride
            x = data[:, idx : idx + self.window_size]
            y = data[:, idx + self.window_size : idx + self.window_size + self.horizon]
            
            if labels == None:
                y_label = None
            else:
                y_label = labels[idx + self.window_size : idx + self.window_size + self.horizon]
                
            window = Data(x=x, edge_idx=None, edge_attr=None, y=y, y_label=y_label, id=id)
            dataset.append(window)
        
        return dataset
    
    def __getitem__(self, idx):
        return self.dataset[idx]     
    
    def __iter__(self):  
        self._idx = 0
        return self
    
    def __next__(self): 
        if self._idx >= self._len:
            raise StopIteration
            
        item = self.dataset[self._idx]
        self._idx += 1
        return item

    def __repr__(self):
        return f'{self.__class__.__name__}(num_nodes={self.num_nodes}, window_size={self.window_size}, stride={self.stride}, horizon={self.horizon})'
            
    def __len__(self):
        return self._len      
