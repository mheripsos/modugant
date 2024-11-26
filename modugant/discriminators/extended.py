from typing import Self, override

from modugant.device import Device
from modugant.discriminators.protocol import Discriminator
from modugant.matrix import Matrix
from modugant.matrix.dim import One


class ExtendedDiscriminator[C: int, D: int](Discriminator[C, D]):
    '''Extend and Override Inheritable Discriminator.'''

    def __init__(self, discriminator: Discriminator[C, D]) -> None:
        '''
        Extend and Override Inheritable Discriminator.

        Args:
            discriminator (Discriminator[C, D]): The discriminator model.

        '''
        self._discriminator = discriminator
        self._conditions = discriminator.conditions
        self._outputs = discriminator.outputs
    @override
    def predict[N: int](self, condition: Matrix[N, C], data: Matrix[N, D]) -> Matrix[N, One]:
        return self._discriminator.predict(condition, data)
    @override
    def loss[N: int](self, condition: Matrix[N, C], data: Matrix[N, D], target: Matrix[N, One]) -> Matrix[One, One]:
        return self._discriminator.loss(condition, data, target)
    @override
    def step[N: int](self, condition: Matrix[N, C], data: Matrix[N, D], target: Matrix[N, One]) -> Matrix[One, One]:
        return self._discriminator.step(condition, data, target)
    @override
    def reset(self) -> None:
        self._discriminator.reset()
    @override
    def restart(self) -> None:
        self._discriminator.restart()
    @override
    def move(self, device: Device) -> Self:
        _ = self._discriminator.move(device)
        return self
    @override
    def train(self, mode: bool = True) -> Self:
        _ = self._discriminator.train(mode)
        return self
    @property
    @override
    def rate(self) -> float:
        return self._discriminator.rate
