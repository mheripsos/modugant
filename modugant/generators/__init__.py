'''Pre-defined GAN generators.'''
from .base import BasicGenerator
from .protocol import Generator
from .residual import ResidualGenerator
from .sequential import SequentialGenerator

__all__ = ['BasicGenerator', 'Generator', 'ResidualGenerator', 'SequentialGenerator']
