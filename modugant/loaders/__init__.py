'''Pre-built Loaders.'''
from .category import CategoryLoader
from .direct import DirectLoader
from .joint import JointLoader
from .standardize import StandardizeLoader

__all__ = [
    'CategoryLoader',
    'DirectLoader',
    'JointLoader',
    'StandardizeLoader'
]
