'''Pre-built Loaders.'''
from .direct import DirectLoader
from .onehot import OneHotLoader
from .pooled import PooledLoader
from .standardize import StandardizeLoader

__all__ = [
    'OneHotLoader',
    'DirectLoader',
    'PooledLoader',
    'StandardizeLoader'
]
