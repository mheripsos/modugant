from typing import Tuple, cast, override

from torch import Tensor, no_grad
from torch.nn import Dropout, Linear, ReLU, Sequential
from torch.optim.adam import Adam

from modugant.discriminators.penalized import ReshapingDiscriminator
from modugant.discriminators.standard import StandardDiscriminator
from modugant.matrix import Matrix
from modugant.matrix.dim import Dim, One
from modugant.matrix.ops import cat


class SphLinear(Linear):
    '''Linear layer with weights on the unit sphere.'''

    @staticmethod
    def project(network: Linear) -> None:
        '''Project the weights of the network onto the unit sphere.'''
        with no_grad():
            norm = cast(Tensor, network.weight.norm(2, dim = 1, keepdim = True))
            _ = network.weight.div_(norm)
    def __init__(self, inputs: int, outputs: int):
        '''
        Initialize the linear layer with weights on the unit sphere.

        Args:
            inputs (int): The number of input nodes.
            outputs (int): The number of output nodes.

        '''
        super().__init__(inputs, outputs)
        SphLinear.project(self)
    def reproject(self) -> None:
        '''Project the weights of the layer onto the unit sphere.'''
        SphLinear.project(self)
    @override
    def reset_parameters(self) -> None:
        '''Re-initialize the weights of the layer.'''
        super().reset_parameters()
        SphLinear.project(self)


## Make this a ReshapedDiscriminator, so that it can be used with our Regularizer
## Just perform a concatenation in the reshape (identity-like)
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
            layer = lambda ins, outs, _: Sequential(
                SphLinear(ins, outs),
                ReLU(),
                Dropout(dropout)
            ),
            finish = lambda ins: Sequential(SphLinear(ins, 1))
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
        return Matrix(data, (n, Dim.one()))
    @override
    def loss[N: int](self, condition: Matrix[N, C], data: Matrix[N, D], target: Matrix[N, One]) -> Matrix[One, One]:
        predicted = self.predict(condition, data)
        loss = - (target.t() @ predicted.log() + (1 - target.t()) @ (1 - predicted).log()) / len(target)
        return loss
    @override
    def step[N: int](self, condition: Matrix[N, C], data: Matrix[N, D], target: Matrix[N, One]) -> Matrix[One, One]:
        loss = super().step(condition, data, target)
        # project all linear weight vectors back onto unit sphere after updates
        for module in self.modules():
            if isinstance(module, SphLinear):
                module.reproject()
        return loss
    @override
    def restart(self) -> None:
        self._optimizer = Adam(
            self.parameters(),
            lr = self.__lr,
            betas = self.__betas,
            weight_decay = self.__decay,
        )
