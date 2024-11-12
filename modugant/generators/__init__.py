'''Pre-defined GAN generators.'''
from .base import BasicGenerator
from .residual import ResidualGenerator
from .sequential import SequentialGenerator

__all__ = ['BasicGenerator', 'ResidualGenerator', 'SequentialGenerator']
