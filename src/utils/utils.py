import torch
import torch.nn.functional as F
from .device import get_device

def normalize_with_median_iqr(x):
    ''' 
    Row normalization with median und interquartile range for 2d tensors.

    Args:
        x (Tensor): 2-dimensional input tensor.
    '''
    assert isinstance(x, torch.Tensor)

    device = get_device()
    
    quantiles = torch.tensor([.25, .5, .75]).to(device)
    q1, median, q3, = torch.quantile(x, quantiles, dim=1)
    iqr = q3 - q1

    return (x - median.unsqueeze(0).T) / (1 + iqr.unsqueeze(0).T)

def weighted_average_smoothing(x, k, mode='mean'):      
    ''' 
    Average (weighted) smooothing of rows of a 2d tensor with 1d kernel, padding='same'.
    
    Args:
        x (Tensor): 2-dimensional input tensor.
        k (int): Size of the smoothing kernel.
        mode (str): Weighting of the average. Can be:
            'mean' : no weighting
            'exp' : exponentially gives heigher weights to the right side of a row

    '''
    assert isinstance(x, torch.Tensor)

    device = get_device()

    n = x.size(0)
    div, mod = divmod(k, 2)
    p1d = (div, div - (mod ^ 1)) # padding size
    x = torch.constant_pad_nd(x, p1d, value=0.0)
    x = x.view(n, 1, -1)

    if mode == 'mean':
        kernel = torch.full(size=(1,1,k), fill_value=1/k, requires_grad=False)
    elif mode == 'exp':
        kernel = torch.logspace(-k+1, 0, k, base=1.5, requires_grad=False)
        kernel /= kernel.sum()
        kernel = kernel.view(1,1,k)
    
    return F.conv1d(x, kernel.to(device)).squeeze()

def cast(dtype, *args):
    ''' 
    Casts arbitrary number of tensors to specified type.

    Args:
        dtype (type or string): The desired type.
        *args: Tensors to be type-cast.

    '''
    a = [x.type(dtype) for x in args]
    if len(a) == 1:
        return a.pop()
    else:
        return a

def equalize_len(t1, t2, value=0):
    ''' 
    Returns new tensors with equal length according to max(len(t1), len(t2)).

    Args:
        t1 (Tensor): Input tensor
        t2 (Tensor): Input tensor
        value (int, float, optional): Fill value for new entries in shorter tensor.
    '''

    assert isinstance(t1, torch.Tensor)
    assert isinstance(t2, torch.Tensor)

    if len(t1) == len(t2):
        return t1, t2
    
    diff = abs(len(t2) - len(t1))
    p1d = (0, diff)
    if len(t1) > len(t2):
        t2 = F.pad(t2, p1d, 'constant', value)
        return t1, t2
    else:
        t1 = F.pad(t1, p1d, 'constant', value)
        return t1, t2

def format_time(t):
    ''' 
    Format seconds to days, hours, minutes, and seconds.
    -> Output format example: "01d-09h-24m-54s"
    
    Args:
        t (float, int): Time in seconds.
    '''
    assert isinstance(t, (float, int))

    h, r = divmod(t,3600)
    d, h = divmod(h, 24)
    m, r = divmod(r, 60)
    s, r = divmod(r, 1)

    values = [d, h, m, s]
    symbols = ['d', 'h', 'm', 's']
    for i, val in enumerate(values):
        if val > 0:
            symbols[i] = ''.join([f'{int(val):02d}', symbols[i]])
        else:
            symbols[i] = ''
    return '-'.join(s for s in symbols if s) if any(symbols) else '<1s'