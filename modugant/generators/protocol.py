from typing import Protocol, Self

from modugant.device import Device
from modugant.matrix.dim import One
from modugant.matrix.matrix import Matrix
from modugant.protocols import WithConditions, WithLatent, WithOutputs


class Generator[C: int, L: int, D: int](WithConditions[C], WithLatent[L], WithOutputs[D], Protocol):
    '''
    Generator for GANs.

    Type parameters:
        C: The number of conditions.
        L: The number of latent inputs.
        D: The number of generated outputs.

    Abstract properties (must be assigned in subclass):
        _conditions: C; The number of conditions.
        _latents: L; The number of latent inputs.
        _outputs: D; The number of generated outputs.

    Abstract methods (must be implemented in subclass):
        sample: Generate a sample from conditions.
            [N:int](condition: Matrix[N, C]) -> Matrix[N, G]
        update: Update the generator with the given loss.
            (loss: Matrix[One, One]) -> None
        reset: Reset the generator.
            () -> None
        restart: Restart the learning rate scheduler.
            () -> None
        rate: The current learning rate of the generator.
            property: () -> float

    '''

    def sample[N: int](self, condition: Matrix[N, C]) -> Matrix[N, D]:
        '''
        Generate a sample from conditions.

        Args:
            condition (Tensor (N, C)): The condition for the sample.

        Returns:
            Tensor (N, D): The generated sample.

        '''
        ...
    def update(self, loss: Matrix[One, One]) -> None:
        '''
        Update the discriminator with the given loss.

        Args:
            loss (Tensor (1, 1)): The loss of the generator.

        '''
        ...
    def reset(self) -> None:
        '''Reset the discriminator.'''
        ...
    def restart(self) -> None:
        '''Restart the learning rate scheduler.'''
        ...
    def move(self, device: Device) -> Self:
        '''Move the generator to the device.'''
        ...
    def train(self, mode: bool) -> Self:
        '''Set the generator to training mode.'''
        ...
    @property
    def rate(self) -> float:
        '''The current learning rate of the generator.'''
        ...
