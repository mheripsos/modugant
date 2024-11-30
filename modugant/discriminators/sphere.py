from typing import Tuple, override

from torch import Tensor
from torch.optim.adam import Adam

from modugant.discriminators.penalized import ReshapingDiscriminator
from modugant.discriminators.standard import StandardDiscriminator
from modugant.layers.isometric.dropout import DropoutLayer
from modugant.layers.isometric.relu import RectifiedLayer
from modugant.layers.linear.sphere import SphericalLayer
from modugant.matrix import Matrix
from modugant.matrix.dim import Dim, One
from modugant.matrix.index import Index
from modugant.matrix.ops import cat


class SphereDiscriminator[C: int, D: int](StandardDiscriminator[C, D], ReshapingDiscriminator[C, D]):
    '''Discriminator model with sphere regularization.'''

    def __init__(
        self,
        conditions: C,
        outputs: D,
        steps: list[int],
        dropout: float = 0.2,
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.5, 0.9),
        decay: float = 0.1,
    ) -> None:
        '''
        Initialize the discriminator model.

        Args:
            conditions (C: int): The number of condition nodes.
            outputs (D: int): The number of input nodes.
            steps (list[int]): The number of nodes in each step of the discriminator.
            dropout (float): The dropout rate.
            lr (float): The learning rate.
            betas (Tuple[float, float]): The beta values for the Adam optimizer.
            decay (float): The weight decay.

        '''
        super().__init__(
            conditions,
            outputs,
            steps,
            layer = lambda ins, outs, _: SphericalLayer(
                outs,
                Index.range(ins),
                [RectifiedLayer(outs), DropoutLayer(outs, dropout)]
            ),
            finish = lambda ins: SphericalLayer(Dim.one(), Index.range(ins))
        )
        self.__lr = lr
        self.__decay = decay
        self.__betas = betas
        self._optimizer = Adam(
            self.parameters(),
            lr = lr,
            betas = betas,
            weight_decay = decay,
        )
    @override
    def reshape[N: int](self, condition: Matrix[N, C], data: Matrix[N, D]) -> Tensor:
        cols = self._conditions + self._outputs
        joined = cat(
            (condition, data),
            dim = 1,
            shape = (data.shape[0], cols)
        )
        return joined
    @override
    def unshape[N: int](self, data: Tensor, n: N) -> Matrix[N, One]:
        return Matrix.cast(data, (n, Dim.one()))
    @override
    def loss[N: int](self, condition: Matrix[N, C], data: Matrix[N, D], target: Matrix[N, One]) -> Matrix[One, One]:
        predicted = self.predict(condition, data)
        loss = - (target.t() @ predicted.log() + (1 - target.t()) @ (1 - predicted).log()) / len(target)
        return loss
    @override
    def restart(self) -> None:
        self._optimizer = Adam(
            self.parameters(),
            lr = self.__lr,
            betas = self.__betas,
            weight_decay = self.__decay,
        )
