'''Pre-built Loaders.'''
from .direct import DirectLoader
from .joint import JointLoader
from .onehot import OneHotLoader
from .standardize import StandardizeLoader

__all__ = [
    'OneHotLoader',
    'DirectLoader',
    'JointLoader',
    'StandardizeLoader'
]
