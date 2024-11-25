from typing import Self

from modugant.device import Device
from modugant.loaders.connectors.protocol import PreConnector
from modugant.matrix.matrix import Matrix


class Loader[S: int, C: int, D: int](PreConnector[S, C, D]):
    '''
    Connector for GANs.

    Type parameters:
        S: The sampled dimension.
        C: The condition dimension.
        D: The data dimension.

    Abstract properties (must be assigned in subclass):
        _samples: S; The number of samples.
        _conditions: C; The number of conditions.
        _outputs: D; The number of transformed outputs.

    Abstract methods (must be implemented in subclass):
        condition: Condition the data.
            [N:int](data: Matrix[N, int]) -> Matrix[N, C]
        prepare: Transform the data based on the condition.
            [N:int](condition: Matrix[N, C], intermediate: Matrix[N, D]) -> Matrix[N, D]
        loss: Calculate the loss of the data source on the given condition and output.
            [N:int](condition: Matrix[N, C], intermediate: Matrix[N, D]) -> Matrix[One, One]
        sample: Sample the data.
            [N:int](batch: N) -> Matrix[N, int]
        load: Encode the data.
            [N:int](data: Matrix[N, int]) -> Matrix[N, D]
        unload: Decode the data.
            [N:int](data: Matrix[N, D]) -> Matrix[N, int]
        restart: Restart the sampler.
            () -> None
        update: Update the data source. (default: pass)
            () -> None
    '''

    def sample[N: int](self, batch: N) -> Matrix[N, S]:
        '''
        Sample the data.

        Args:
            batch (N: int): The batch size.

        Returns:
            Matrix[N, S]: The encoded data.

        '''
        ...
    def move(self, device: Device) -> Self:
        '''Move the connector to the device.'''
        ...
    def restart(self) -> None:
        '''Restart the sampler.'''
        ...
