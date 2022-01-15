import torch

# Dirty file that holds the device (cpu or cuda).
# Import setter or getter functions to access.
# get_device determine the device from availability
# if none is set manually. 

_device = None 

def get_device():
    ''' Returns the currently used computing device.'''
    if _device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        set_device(device)
    return _device

def set_device(device):
    ''' Sets the computing device. '''
    global _device
    _device = device
