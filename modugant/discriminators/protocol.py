from typing import Protocol, Self

from modugant.device import Device
from modugant.matrix.dim import One
from modugant.matrix.matrix import Matrix
from modugant.protocols import WithConditions, WithOutputs


class Discriminator[C: int, D: int](WithConditions[C], WithOutputs[D], Protocol):
    '''
    Discriminator for GANs.

    Type parameters:
        C: The number of conditions.
        D: The number of inputs of data.

    Abstract properties (must be assigned in subclass):
        _conditions: C; The number of conditions.
        _outputs: D; The number of inputs of data.

    Abstract methods (must be implemented in subclass):
        predict: Pass the data through the discriminator.
            [N:int](condition: Matrix[N, C], data: Matrix[N, D]) -> Matrix[N, One]
        loss: Calculate the loss of the discriminator on the given predictions and target.
            [N:int](condition: Matrix[N, C], data: Matrix[N, D], target: Matrix[N, One]) -> Matrix[One, One]
        step: Update the discriminator on the given data and target. Return the loss.
            [N:int](condition: Matrix[N, C], data: Matrix[N, D], target: Matrix[N, One]) -> Matrix[One, One]
        reset: Reset the discriminator parameters.
            () -> None
        restart: Restart the discriminators state
            () -> None
        rate: The current learning rate of the generator.
            property: () -> float

    '''

    def predict[N: int](self, condition: Matrix[N, C], data: Matrix[N, D]) -> Matrix[N, One]:
        '''
        Pass the data through the discriminator.

        Args:
            condition (Tensor (N, C)): The condition for the data.
            data (Tensor (N, D)): The data to pass through the
                discriminator.

        Returns:
            Tensor (N, 1): The output of the discriminator.

        '''
        ...
    def loss[N: int](self, condition: Matrix[N, C], data: Matrix[N, D], target: Matrix[N, One]) -> Matrix[One, One]:
        '''
        Calculate the loss of the discriminator on the given predictions and target.

        Args:
            condition (Tensor (N, C)): The condition for the data.
            data (Tensor (N, D)): The data to pass through the discriminator.
            target (Tensor (N, 1)): The target of the

        Returns:
            Tensor (1, 1): The loss of the discriminator.

        '''
        ...
    def step[N: int](self, condition: Matrix[N, C], data: Matrix[N, D], target: Matrix[N, One]) -> Matrix[One, One]:
        '''
        Update the discriminator on the given data and target. Return the loss.

        Args:
            condition (Tensor (N, C)): The condition for the data.
            data (Tensor (N, D)): The data to pass through the discriminator.
            target (Tensor (N, 1)): The target of the discriminator

        Returns:
            Tensor (1, 1): The loss of the discriminator.

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
