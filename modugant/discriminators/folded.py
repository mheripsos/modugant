'''
Re-implementation of the Discriminator class inspired by the CTGAN library.

CTGAN (Conditional Tabular GAN) is a library developed by the SDV (Synthetic Data Vault) team,
which provides tools for generating synthetic tabular data using GANs.
The original implementation can be found at: https://github.com/sdv-dev/CTGAN

This re-implementation aims to replicate and extend the functionality of the CTGAN Discriminator class,
with modifications to suit specific better modularity.

Classes:
    ChunkedDiscriminator: A neural network model designed to distinguish between real and synthetic data samples.

Usage:
    discriminator = Discriminator(input_dim, hidden_dim)
    output = discriminator(input_data)

Attributes:
    input_dim (int): The dimensionality of the input data.
    hidden_dim (int): The dimensionality of the hidden layers.

'''

from typing import Tuple, override

from torch import Tensor, cat
from torch.nn import Dropout, LeakyReLU, Linear, Sequential
from torch.optim.adam import Adam

from modugant.discriminators.penalized import ReshapingDiscriminator
from modugant.discriminators.standard import StandardDiscriminator
from modugant.matrix import Matrix
from modugant.matrix.dim import Dim, One


class FoldedDiscriminator[C: int, D: int](StandardDiscriminator[C, D], ReshapingDiscriminator[C, D]):
    '''Discriminator model for GANs.'''

    def __init__(
        self,
        conditions: C,
        outputs: D,
        group: int,
        steps: list[int],
        dropout: float = 0.5,
        lr: float = 0.1,
        betas: Tuple[float, float] = (0.5, 0.9),
        decay: float = 0.1,
        slope: float = 0.1
    ) -> None:
        '''
        Initialize the discriminator model.

        Args:
            conditions (C: int): The number of condition nodes.
            outputs (D: int): The number of input nodes.
            group (int): The number of groups to split the input nodes into.
            steps (list[int]): The number of nodes in each step of the discriminator.
            dropout (float): The dropout rate.
            lr (float): The learning rate.
            betas (Tuple[float, float]): The beta values for the Adam optimizer.
            decay (float): The weight decay.
            slope (float): The slope of the LeakyReLU activation function.

        '''
        super().__init__(
            conditions,
            outputs,
            steps,
            layer = lambda ins, outs, layer: Sequential(
                Linear(
                    ins if layer else group * (outputs + conditions),
                    outs
                ),
                LeakyReLU(slope),
                Dropout(dropout)
            ),
            finish = lambda ins: Linear(ins, 1)
        )
        self.__group = group
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
        assert data.shape[0] % self.__group == 0, "Data must be divisible by the group size"
        joined = cat([condition, data], dim = 1) # join the condition and data
        return joined.view(-1, self.__group * (self._conditions + self._outputs))
    @override
    def unshape[N: int](self, data: Tensor, n: N) -> Matrix[N, One]:
        replicated = data.expand(-1, self.__group).reshape(-1, 1)
        return Matrix.cast(replicated, shape = (n, Dim.one()))
    @override
    def loss[N: int](self, condition: Matrix[N, C], data: Matrix[N, D], target: Matrix[N, One]) -> Matrix[One, One]:
        predicted = self.predict(condition, data)
        weight = target.sum(dim = 0, keepdim = True) / len(target)
        loss = (
            ((1 - weight) * (1 - target.t())) @ predicted -
            (weight * target.t()) @ predicted
        )
        return loss
    @override
    def restart(self) -> None:
        self._optimizer = Adam(
            self.parameters(),
            lr = self.__lr,
            betas = self.__betas,
            weight_decay = self.__decay
        )
