import torch
import torch.nn.functional as F

# Transforms applied to datasets, e.g. for downsampling to speed up training

class BaseSampling(torch.nn.Module):
    '''
    Base class for sampling transforms applied to tensors.

    Args:
        k (int): Number of samples to be aggregated.
    '''
    def __init__(self, k):
        super().__init__()

        self._k = k

    def forward(self, x):
        dims = len(x.shape)
        p1d = 0, (self._k - len(x) % self._k) % self._k
        x = F.pad(x, p1d, "constant", 0)
        x = x.unfold(dims-1, self._k, self._k)
        x = self.sampling(x)
        return x 

    def sampling(self, x):
        raise NotImplementedError

class MedianSampling2d(BaseSampling):
    '''
    Returns a 2d tensor where each row is downsampled with the median of k values.
    Only for 2d tensors.
    '''
    def __init__(self, k):
        super().__init__(k)

    def sampling(self, x): 
        assert len(x.shape) == 3
        x, _ = x.median(dim=2)
        return x

class MedianSampling1d(BaseSampling):
    '''
    Returns a 1d tensor that is downsampled with the median of k values.
    Only for 1d tensors.
    '''
    def __init__(self, k):
        super().__init__(k)

    def sampling(self, x): 
        assert len(x.shape) == 2
        x, _ = x.median(dim=1)
        return x

class MaxSampling1d(BaseSampling):
    '''
    Returns a 1d tensor that is downsampled with the maximum of k values.
    Only for 1d tensors.
    '''
    def __init__(self, k):
        super().__init__(k)

    def sampling(self, x):
        assert len(x.shape) == 2
        x, _ = x.max(dim=1)
        return x 

