from typing import override

from modugant.matrix import Matrix
from modugant.protocols import Loader, Sampler


class LoadingSampler[D: int](Sampler[D]):
    '''Loading sampler for GANs.'''

    def __init__(
        self,
        sampler: Sampler[int],
        loader: Loader[D]
    ) -> None:
        '''
        Initialize the compiled sampler.

        Args:
            sampler (Sampler): The sampler.
            loader (Loader): The loader.

        '''
        self._sampler = sampler
        self._loader = loader
        self._outputs = loader.outputs
    @override
    def sample[N: int](self, batch: N) -> Matrix[N, D]:
        return self._loader.load(self._sampler.sample(batch))
    @override
    def restart(self) -> None:
        self._sampler.restart()
    @property
    @override
    def holdout(self) -> Matrix[int, D]:
        return self._loader.load(self._sampler.holdout)
