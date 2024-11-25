'''
Protocol classes for GANs.

Protocols:
    Generator: Protocol for GAN generator.
    Discriminator: Protocol for GAN discriminator.
    Sampler: Protocol for GAN sampler.
    Regimen: Protocol for GAN training regimen.
'''

from typing import Protocol


class WithLatent[L: int](Protocol):
    '''Abstract class with latent property.'''

    _latents: L
    @property
    def latents(self) -> L:
        '''The latent dimension.'''
        return self._latents

class WithSamples[S: int](Protocol):
    '''Abstract class with samples property.'''

    _samples: S
    @property
    def samples(self) -> S:
        '''The sampled dimension.'''
        return self._samples

class WithConditions[C: int](Protocol):
    '''Abstract class with conditions property.'''

    _conditions: C
    @property
    def conditions(self) -> C:
        '''The condition dimension.'''
        return self._conditions

class WithOutputs[D: int](Protocol):
    '''Abstract class with outputs property.'''

    _outputs: D
    @property
    def outputs(self) -> D:
        '''The data.'''
        return self._outputs

class Updatable(Protocol):
    '''Protocol for updatable classes.'''

    def update(self) -> None:
        '''Update the data source.'''
        pass
