'''Pre-built Penalizers.'''
from .entropy import EntropyPenalizer
from .joint import JointPenalizer
from .pooled import PooledPenalizer
from .static import StaticPenalizer

__all__ = [
    'EntropyPenalizer',
    'JointPenalizer',
    'PooledPenalizer',
    'StaticPenalizer',
]
