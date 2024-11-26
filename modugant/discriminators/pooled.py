from typing import Self, Sequence, override

from modugant.device import Device
from modugant.discriminators.basic import BasicDiscriminator
from modugant.discriminators.protocol import Discriminator
from modugant.matrix import Matrix
from modugant.matrix.dim import One


class PooledDiscriminator[C: int, D: int](Discriminator[C, D]):
    '''A Discriminator that adds a pool of losses and updates to a main Discriminator.'''

    def __init__(
        self,
        main: BasicDiscriminator[C, D],
        pool: Sequence[BasicDiscriminator[C, D]],
    ) -> None:
        '''
        Initialize the pooled discriminator with a main and pool of discriminators.

        Args:
            main (BasicDiscriminator[C, D]): The main discriminator.
            pool (Sequence[BasicDiscriminator[C, D]]): The pool of discriminators.

        '''
        self._main = main
        self._pool = pool
    @override
    def predict[N: int](self, condition: Matrix[N, C], data: Matrix[N, D]) -> Matrix[N, One]:
        return self._main.predict(condition, data)
    @override
    def loss[N: int](self, condition: Matrix[N, C], data: Matrix[N, D], target: Matrix[N, One]) -> Matrix[One, One]:
        main = self._main.loss(condition, data, target)
        pool = sum(d.loss(condition, data, target) for d in self._pool)
        return main + pool
    @override
    def step[N: int](self, condition: Matrix[N, C], data: Matrix[N, D], target: Matrix[N, One]) -> Matrix[One, One]:
        self._main.zero_grad()
        for discriminator in self._pool:
            discriminator.zero_grad()
        loss = self.loss(condition, data, target)
        _ = loss.backward()
        self._main.optimizer.step()
        for discriminator in self._pool:
            discriminator.optimizer.step()
        return loss
    @override
    def reset(self) -> None:
        self._main.reset()
        for discriminator in self._pool:
            discriminator.reset()
    @override
    def restart(self) -> None:
        self._main.restart()
        for discriminator in self._pool:
            discriminator.restart()
    @override
    def move(self, device: Device) -> Self:
        _ = self._main.move(device)
        for discriminator in self._pool:
            _ = discriminator.move(device)
        return self
    @override
    def train(self, mode: bool) -> Self:
        _ = self._main.train(mode)
        for discriminator in self._pool:
            _ = discriminator.train(mode)
        return self
    @property
    @override
    def rate(self) -> float:
        return self._main.rate
