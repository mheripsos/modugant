from typing import Optional, override

from modugant.matrix import Matrix
from modugant.protocols import Conditioner, Connector, Inteceptor, Loader, Penalizer, Sampler
from modugant.samplers import LoadingSampler
from modugant.transformers import ComposedTransformer


class ComposedConnector[C: int, G: int, D: int](ComposedTransformer[C, G, D], Connector[C, G, D]):
    '''Composed connector for GANs.'''

    def __init__(
        self,
        conditioner: Conditioner[C, D],
        interceptor: Inteceptor[C, G, D],
        penalizer: Penalizer[C, G],
        sampler: Sampler[int],
        loader: Optional[Loader[D]] = None
    ) -> None:
        '''
        Initialize the compiled connector.

        Args:
            conditioner (Conditioner): The conditioner.
            interceptor (Inteceptor): The inteceptor.
            penalizer (Penalizer): The penalizer.
            sampler (Sampler): The sampler.
            loader (Optional[Loader]): The loader.

        '''
        super().__init__(conditioner, interceptor, penalizer, loader)
        self._sampler = LoadingSampler(sampler, self._loader)
    @override
    def sample[N: int](self, batch: N) -> Matrix[N, D]:
        return self._sampler.sample(batch)
    @override
    def restart(self) -> None:
        self._sampler.restart()
    @property
    @override
    def holdout(self) -> Matrix[int, D]:
        return self._sampler.holdout
