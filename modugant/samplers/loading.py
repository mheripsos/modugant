from typing import override

from modugant.matrix import Matrix
from modugant.protocols import Loader, Sampler


class LoadingSampler[U: int, S: int](Sampler[S]):
    '''Loading sampler for GANs.'''

    def __init__(
        self,
        sampler: Sampler[U],
        loader: Loader[U, S]
    ) -> None:
        '''
        Initialize the compiled sampler.

        Args:
            sampler (Sampler): The sampler.
            loader (Loader): The loader.

        '''
        self._sampled = loader.outputs
        self._sampler = sampler
        self._loader = loader
    @override
    def sample[N: int](self, batch: N) -> Matrix[N, S]:
        return self._loader.load(self._sampler.sample(batch))
    @override
    def restart(self) -> None:
        self._sampler.restart()
    @property
    @override
    def holdout(self) -> Matrix[int, S]:
        return self._loader.load(self._sampler.holdout)
