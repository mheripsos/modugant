from typing import Optional, Self, override

from modugant.device import Device
from modugant.loaders import DirectLoader
from modugant.matrix import Matrix
from modugant.matrix.dim import One
from modugant.protocols import Conditioner, Inteceptor, Loader, Transformer, Updater
from modugant.updaters import StaticUpdater


class ComposedTransformer[C: int, G: int, D: int](Transformer[C, G, D]):
    '''Composed transformer for GANs.'''

    def __init__(
        self,
        conditioner: Conditioner[C, D],
        interceptor: Inteceptor[C, G, D],
        updater: Optional[Updater[C, G]] = None,
        loader: Optional[Loader[D]] = None,
        device: Device = 'cpu'
    ) -> None:
        '''
        Initialize the compiled transformer.

        Args:
            conditioner (Conditioner[C, D]): The conditioner.
            interceptor (Inteceptor[C, G, D]): The inteceptor.
            updater (Optional[Updater[C, G]]): The updater.
            loader (Optional[Loader[D]]): The loader.
            device (Device): The device.

        '''
        self._conditioner = conditioner
        self._inteceptor = interceptor
        self._updater = updater or StaticUpdater(interceptor.conditions, interceptor.intermediates)
        self._loader = loader or DirectLoader(
            interceptor.outputs,
            [(0, interceptor.outputs)]
        )
        self._conditions = conditioner.conditions
        self._intermediates = interceptor.intermediates
        self._outputs = interceptor.outputs
        self._device = device
    @override
    def condition[N: int](self, data: Matrix[N, D]) -> Matrix[N, C]:
        return self._conditioner.condition(data).to(self._device)
    @override
    def prepare[N: int](self, condition: Matrix[N, C], intermediate: Matrix[N, G]) -> Matrix[N, D]:
        return self._inteceptor.prepare(condition, intermediate).to(self._device)
    @override
    def loss[N: int](self, condition: Matrix[N, C], intermediate: Matrix[N, G]) -> Matrix[One, One]:
        return self._updater.loss(condition, intermediate).to(self._device)
    @override
    def update(self) -> None:
        self._updater.update()
    @override
    def load[N: int](self, data: Matrix[N, int]) -> Matrix[N, D]:
        return self._loader.load(data).to(self._device)
    def move(self, device: Device) -> Self:
        '''Move the transformer to the device.'''
        self._device = device
        return self
