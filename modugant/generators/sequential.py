'''
Straight Generator.

Classes:
    StraightGenerator: Generator model for GANs.

'''

from typing import override

from torch.nn import Sequential
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import StepLR

from modugant.generators.base import BasicGenerator
from modugant.layers.isometric.relu import RectifiedLayer
from modugant.layers.isometric.sigmoid import SigmoidLayer
from modugant.layers.linear.linear import LinearLayer
from modugant.matrix import Matrix
from modugant.matrix.dim import One
from modugant.matrix.index import Index
from modugant.matrix.ops import randn


class SequentialGenerator[C: int, L: int, G: int](BasicGenerator[C, L, G]):
    '''Generator model for GANs.'''

    def __init__(
        self,
        conditions: C,
        latents: L,
        intermediates: G,
        steps: list[int],
        learning: float = 0.1,
        gamma: float = 0.99,
        step: int = 100
    ) -> None:
        '''
        Initialize the generator model.

        Args:
            conditions (C: int): The number of condition nodes.
            latents (L: int): The number of input nodes.
            intermediates (G: int): The number of output nodes.
            steps (list[int]): The number of nodes in each step of the generator.
            learning (float): The learning rate.
            gamma (float): The gamma value for the learning rate scheduler.
            step (int): The step size for the learning rate scheduler.

        '''
        super().__init__(
            conditions,
            latents,
            intermediates
        )
        self._steps = steps
        self._learning = learning
        self._gamma = gamma
        self._step = step
        self._model = Sequential(
            *[
                LinearLayer(
                    steps[i],
                    Index.range(steps[i - 1] if i else (conditions + latents)),
                    [RectifiedLayer(steps[i])]
                ) for i in range(len(steps))
            ],
            LinearLayer(
                intermediates,
                Index.range(steps[-1] if len(steps) else (conditions + latents)),
                [SigmoidLayer(intermediates)]
            )
        )
        self._optimizer = Adam(self.parameters(), lr = learning)
        self.__scheduler = StepLR(self._optimizer, step_size = step, gamma = gamma)
    @override
    def _latent[N: int](self, batch: N) -> Matrix[N, L]:
        return randn((batch, self._latents))
    @override
    def update(self, loss: Matrix[One, One]) -> None:
        super().update(loss)
        self.__scheduler.step()
    @override
    def restart(self) -> None:
        self._optimizer = Adam(self.parameters(), lr = self._learning)
        self.__scheduler = StepLR(self._optimizer, step_size = self._step, gamma = self._gamma)
