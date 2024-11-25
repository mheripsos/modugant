from modugant.matrix.index import Index
from modugant.protocols import Updatable


class Sampler(Updatable):
    '''
    Index sampler for Loader composition.

    Abstract methods (must be implemented in subclass):
        sample: Sample the data.
            [N:int](batch: N) -> Index[N, int]
        restart: Restart the sampler.
            () -> None
    '''

    def sample[N: int](self, batch: N) -> Index[N, int]:
        '''
        Sample the data.

        Args:
            batch (N: int): The batch size.

        Returns:
            Index[N, S]: The sampled data.

        '''
        ...
    def restart(self) -> None:
        '''Restart the sampler.'''
        ...

