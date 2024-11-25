'''Pre-defined discriminator classes for GANs.'''
from .basic import BasicDiscriminator
from .extended import ExtendedDiscriminator
from .folded import FoldedDiscriminator
from .penalized import ReshapingDiscriminator, SmoothedDiscriminator
from .protocol import Discriminator
from .sphere import SphereDiscriminator
from .standard import StandardDiscriminator

__all__ = [
    'BasicDiscriminator',
    'Discriminator',
    'FoldedDiscriminator',
    'ExtendedDiscriminator',
    'ReshapingDiscriminator',
    'SmoothedDiscriminator',
    'SphereDiscriminator',
    'StandardDiscriminator'
]
