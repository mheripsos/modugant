'''
Residual generator for GANs.

Classes:
    ResidualLayer: Residual layer for a generator.
    ResidualGenerator: Generator model for GANs.
'''
from typing import override

from torch.nn import Sequential
from torch.optim.adam import Adam

from modugant.generators.base import BasicGenerator
from modugant.layers.isometric.norm import BatchNormLayer
from modugant.layers.isometric.relu import RectifiedLayer
from modugant.layers.linear.linear import LinearLayer
from modugant.layers.residual import ResidualLayer
from modugant.matrix import Matrix
from modugant.matrix.index import Index
from modugant.matrix.ops import normal


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
        cumul = [conditions + latents + sum(steps[:i]) for i in range(len(steps))]
        self._learning = learning
        self._decay = decay
        self._model = Sequential(
            *[
                ResidualLayer(
                    cumul[i],
                    Index.range(cumul[i]),
                    LinearLayer(
                        steps[i],
                        Index.range(cumul[i]),
                        [BatchNormLayer(steps[i]), RectifiedLayer(steps[i])]
                    )
                )
                for i in range(len(steps))
            ],
            LinearLayer(intermediates, Index.range(sum(steps) + conditions + latents))
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
