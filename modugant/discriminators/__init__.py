'''Pre-defined discriminator classes for GANs.'''
from .basic import BasicDiscriminator
from .extended import ExtendedDiscriminator
from .folded import FoldedDiscriminator
from .penalized import ReshapingDiscriminator, SmoothedDiscriminator
from .sphere import SphereDiscriminator
from .standard import StandardDiscriminator

__all__ = [
    'BasicDiscriminator',
    'FoldedDiscriminator',
    'ExtendedDiscriminator',
    'ReshapingDiscriminator',
    'SmoothedDiscriminator',
    'SphereDiscriminator',
    'StandardDiscriminator'
]
