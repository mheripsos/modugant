from typing import Self, override

from torch import Tensor, cat
from torch.nn import Linear, Module
from torch.optim import Optimizer

from modugant.device import Device
from modugant.discriminators.protocol import Discriminator
from modugant.matrix import Matrix
from modugant.matrix.dim import Dim, One


class BasicDiscriminator[C: int, D: int](Module, Discriminator[C, D]):
    '''
    Basic discriminator model for GANs.

    Type Parameters:
        C: The number of condition nodes.
        D: The dimensionality of the data.

    Abstract Properties (must be assigned by subclasses):
        _model (Module): The model to use.
        _optimizer (Optimizer): The optimizer to use.

    Abstract Methods (must be implemented by subclasses):
        loss: Calculate the loss of the discriminator.
        restart: Restart the optimizer, etc.

    '''

    _model: Module
    _optimizer: Optimizer

    def __init__(
        self,
        conditions: C,
        outputs: D
    ) -> None:
        '''
        Initialize the discriminator model.

        Args:
            conditions (C): The number of condition nodes.
            outputs (D): The number of output nodes.
            model (nn.Module): The model to use.

        '''
        super().__init__()
        self._conditions = conditions
        self._outputs = outputs
    @override
    def forward(self, data: Tensor) -> Tensor:
        return self._model(data)
    @override
    def predict[N: int](self, condition: Matrix[N, C], data: Matrix[N, D]) -> Matrix[N, One]:
        return Matrix.cast(
            self.forward(cat([condition, data], dim = 1)),
            shape = (condition.shape[0], Dim.one())
        )
    @override
    def step[N: int](self, condition: Matrix[N, C], data: Matrix[N, D], target: Matrix[N, One]) -> Matrix[One, One]:
        self.zero_grad()
        loss = self.loss(condition, data, target)
        _ = loss.backward()
        self._optimizer.step()
        return loss
    @override
    def reset(self) -> None:
        for module in self.modules():
            if isinstance(module, Linear):
                module.reset_parameters()
    @override
    def move(self, device: Device) -> Self:
        return self.to(device)
    @override
    def train(self, mode: bool = True) -> Self:
        return super().train(mode)
    @property
    def model(self) -> Module:
        '''The torch Module for the discriminator.'''
        return self._model
    @property
    def optimizer(self) -> Optimizer:
        '''The torch Optimizer for the discriminator.'''
        return self._optimizer
    @property
    @override
    def rate(self) -> float:
        return self._optimizer.param_groups[0]['lr']

