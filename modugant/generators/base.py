'''Basic generator model for GANs.'''
from abc import abstractmethod
from typing import Self, override

from torch import Tensor, no_grad
from torch.nn import Linear, Module
from torch.optim import Optimizer

from modugant.device import Device, check_device
from modugant.generators.protocol import Generator
from modugant.matrix import Matrix
from modugant.matrix.dim import One
from modugant.matrix.ops import cat


class BasicGenerator[C: int, L: int, G: int](Module, Generator[C, L, G]):
    '''
    Abstract class for basic torch.Module generator model for GANs.

        Abstract properties:
            _model: Callable[[Tensor], Tensor]; The generator model.
            _optimizer: Optimizer; The optimizer for the generator.
        Abstract methods:
            _latent: (int) -> Tensor; Generate a latent input from conditions
            restart: () -> None; Set optimizer to initial state.
            clone: () -> Self; Clone the generator model.

    '''

    _model: Module
    _optimizer: Optimizer
    def __init__(
        self,
        conditions: C,
        latents: L,
        intermediates: G
    ):
        '''
        Initialize the generator model.

        Args:
            conditions (C: int): The number of condition nodes.
            latents (L: int): The number of input nodes.
            intermediates (G: int): The number of output nodes.
            optimizer (Optimizer): The optimizer for the generator.
            device (Device): The device to use.

        '''
        super().__init__()
        self._conditions = conditions
        self._latents = latents
        self._intermediates = intermediates
    @abstractmethod
    def _latent[N: int](self, batch: N) -> Matrix[N, L]:
        '''
        Generate a latent input from conditions.

        Args:
            batch (N: int): The number of latent inputs to generate.

        Returns:
            Tensor (N, C): The latent input.

        '''
        ...
    @override
    def forward(self, data: Tensor) -> Tensor:
        return self._model(data)
    @override
    def sample[N: int](self, condition: Matrix[N, C]) -> Matrix[N, G]:
        latent = self._latent(condition.shape[0])
        inputs = cat((condition, latent), dim = 1, shape = (condition.shape[0], self._conditions + self._latents))
        return Matrix(self.forward(inputs), shape = (condition.shape[0], self._intermediates))
    @override
    def update(self, loss: Matrix[One, One]) -> None:
        self._optimizer.zero_grad()
        _ = loss.backward()
        self._optimizer.step()
    @override
    def reset(self) -> None:
        with no_grad():
            for module in self.modules():
                if isinstance(module, Linear):
                    module.reset_parameters()
    @override
    def move(self, device: Device) -> Self:
        return self.to(check_device(device))
    @override
    def train(self, mode: bool = True) -> Self:
        return super().train(mode)
    @property
    @override
    def rate(self) -> float:
        return self._optimizer.param_groups[0]['lr']
