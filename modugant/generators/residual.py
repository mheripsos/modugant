'''
Residual generator for GANs.

Classes:
    ResidualLayer: Residual layer for a generator.
    ResidualGenerator: Generator model for GANs.
'''
from typing import override

from torch import Tensor, cat
from torch.nn import BatchNorm1d, Linear, Module, ReLU, Sequential
from torch.optim.adam import Adam

from modugant.generators.base import BasicGenerator
from modugant.matrix import Matrix
from modugant.matrix.ops import normal


class ResidualLayer(Module):
    '''Residual layer for a generator.'''

    def __init__(self, inputs: int, outputs: int) -> None:
        '''
        Initialize the residual layer.

        Args:
            inputs (int): The number of input nodes.
            outputs (int): The number of output nodes.

        '''
        super(ResidualLayer, self).__init__()
        self.__model = Sequential(
            Linear(inputs, outputs),
            BatchNorm1d(outputs),
            ReLU()
        )
    @override
    def forward(self, data: Tensor) -> Tensor:
        return cat([self.__model(data), data], dim = 1)

class ResidualGenerator[C: int, L: int, G: int](BasicGenerator[C, L, G]):
    '''Generator model for GANs.'''

    def __init__(
        self,
        conditions: C,
        latents: L,
        intermediates: G,
        steps: list[int],
        learning: float = 0.1,
        decay: float = 0
    ) -> None:
        '''
        Initialize the generator model.

        Args:
            conditions (C: int): The number of condition nodes.
            latents (L: int): The number of input nodes.
            intermediates (G: int): The number of output nodes.
            steps (list[int]): The number of nodes in each step of the generator.
            learning (float): The learning rate.
            decay (float): The weight decay

        '''
        super().__init__(
            conditions,
            latents,
            intermediates
        )
        self._steps = steps
        self._learning = learning
        self._decay = decay
        self._model = Sequential(
            *[
                ResidualLayer(
                    sum(steps[:i]) + latents + conditions,
                    steps[i]
                )
                for i in range(len(steps))
            ],
            Linear(sum(steps) + latents + conditions, intermediates)
        )
        self._optimizer = Adam(
            self.parameters(),
            lr = learning,
            betas = (0.5, 0.9),
            weight_decay = decay
        )
    @override
    def _latent[N: int](self, batch: N) -> Matrix[N, L]:
        return normal(0, 1, (batch, self._latents))
    @override
    def restart(self) -> None:
        self._optimizer = Adam(
            self.parameters(),
            lr = self._learning,
            betas = (0.5, 0.9),
            weight_decay = self._decay
        )
