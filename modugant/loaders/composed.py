from typing import Self, override

from modugant.device import Device, check_device
from modugant.loaders.connectors.composed import ComposedPreConnector
from modugant.loaders.connectors.protocol import Connector
from modugant.loaders.protocol import Loader
from modugant.loaders.samplers.protocol import Sampler
from modugant.matrix.matrix import Matrix


class ComposedLoader[S: int, C: int, D: int](ComposedPreConnector[S, C, D], Loader[S, C, D]):
    '''
    Composed loader for GANs.

    Type parameters:
        S: The number of data inputs.
        C: The number of conditions.
        D: The number of data inputs.

    '''

    def __init__(
        self,
        data: Matrix[int, int],
        sampler: Sampler,
        connector: Connector[S, C, D],
        device: Device = 'cpu'
    ) -> None:
        '''
        Initialize the composed loader.

        Args:
            data (Matrix): The data.
            sampler (Sampler): The sampler.
            connector (Connector): The connector.
            device (Device): The device.

        '''
        super().__init__(connector, connector)
        self.__data = data
        self._sampler = sampler
        self._connector = connector
        self._device = check_device(device)
    @override
    def sample[N: int](self, batch: N) -> Matrix[N, S]:
        '''Sample the data.'''
        sample = self._sampler.sample(batch)
        transformed = self._connector.load(self.__data[sample, ...])
        return transformed.to(self._device)
    @override
    def restart(self) -> None:
        '''Restart the loader.'''
        self._sampler.restart()
    @override
    def move(self, device: Device) -> Self:
        '''Move the transformer to the device.'''
        self._device = device
        return self
    @override
    def update(self) -> None:
        '''Update the loader.'''
        super().update()
        self._sampler.update()
