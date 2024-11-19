from typing import Optional, Self, override

from modugant.device import Device
from modugant.loaders import DirectLoader
from modugant.matrix import Matrix
from modugant.matrix.dim import One
from modugant.matrix.index import Index
from modugant.penalizers import StaticPenalizer
from modugant.protocols import Conditioner, Inteceptor, Loader, Penalizer, Transformer


class ComposedTransformer[S: int, C: int, G: int, D: int](Transformer[S, C, G, D]):
    '''Composed transformer for GANs.'''

    def __init__(
        self,
        conditioner: Conditioner[S, C],
        interceptor: Inteceptor[C, G, D],
        penalizer: Optional[Penalizer[C, G]] = None,
        loader: Optional[Loader[S, D]] = None,
        device: Device = 'cpu'
    ) -> None:
        '''
        Initialize the compiled transformer.

        Args:
            conditioner (Conditioner[C, D]): The conditioner.
            interceptor (Inteceptor[C, G, D]): The inteceptor.
            penalizer (Optional[Penalizer[C, G]]): The penalizer.
            loader (Optional[Loader[D]]): The loader.
            device (Device): The device.

        '''
        self._conditioner = conditioner
        self._inteceptor = interceptor
        self._penalizer = penalizer or StaticPenalizer(interceptor.conditions, interceptor.intermediates)
        self._loader = loader or DirectLoader(
            Index.slice(0, interceptor.outputs, conditioner.sampled)
        )
        self._conditions = conditioner.conditions
        self._intermediates = interceptor.intermediates
        self._outputs = interceptor.outputs
        self._device = device
    @override
    def condition[N: int](self, data: Matrix[N, S]) -> Matrix[N, C]:
        return self._conditioner.condition(data).to(self._device)
    @override
    def prepare[N: int](self, condition: Matrix[N, C], intermediate: Matrix[N, G]) -> Matrix[N, D]:
        return self._inteceptor.prepare(condition, intermediate).to(self._device)
    @override
    def loss[N: int](self, condition: Matrix[N, C], intermediate: Matrix[N, G]) -> Matrix[One, One]:
        return self._penalizer.loss(condition, intermediate).to(self._device)
    @override
    def update(self) -> None:
        self._conditioner.update()
        self._inteceptor.update()
        self._penalizer.update()
        self._loader.update()
    @override
    def load[N: int](self, data: Matrix[N, S]) -> Matrix[N, D]:
        return self._loader.load(data).to(self._device)
    @override
    def unload[N: int](self, data: Matrix[N, D]) -> Matrix[N, int]:
        return self._loader.unload(data)
    def move(self, device: Device) -> Self:
        '''Move the transformer to the device.'''
        self._device = device
        return self
