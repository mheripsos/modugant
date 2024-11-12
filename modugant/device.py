from os import environ
from typing import Callable, Literal

from torch import device
from torch.backends.mkldnn import is_available as mkl_is_available
from torch.backends.mps import is_available as mps_is_available
from torch.cuda import is_available as cuda_is_available
from torch.xpu import is_available as xpu_is_available

type Device = Literal['cpu', 'cuda', 'mkldnn', 'mps', 'xpu', 'xla']

def ondevice[**A, R](key: Device) -> Callable[[Callable[A, R]], Callable[A, R]]:
    '''Create a decorator to run a function on a device.'''
    def decorator(function: Callable[A, R]) -> Callable[A, R]:
        '''Decorate the function to run on a device.'''
        def wrapper(*args: A.args, **kwargs: A.kwargs) -> R:
            '''Wrap the function to run on a device.'''
            with device(key):
                return function(*args, **kwargs)
        return wrapper
    return decorator

def check_device(key: Device) -> Device:
        '''Check if device is available, if so, return it. If not, return 'cpu'.'''
        if key == 'cuda' and not cuda_is_available():
            return 'cpu'
        if key == 'mkldnn' and not mkl_is_available():
            return 'cpu'
        if key == 'mps' and not mps_is_available():
            return 'cpu'
        if key == 'xla' and 'COLAB_TPU_ADDR' not in environ:
            return 'cpu'
        if key == 'xpu' and not xpu_is_available():
            return 'cpu'
        return key
