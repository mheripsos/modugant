from typing import Optional, override

from modugant.matrix import Matrix
from modugant.protocols import Conditioner, Connector, Inteceptor, Loader, Penalizer, Sampler
from modugant.transformers import ComposedTransformer


class ComposedConnector[S: int, C: int, G: int, D: int](ComposedTransformer[S, C, G, D], Connector[S, C, G, D]):
    '''Composed connector for GANs.'''

    def __init__(
        self,
        conditioner: Conditioner[S, C],
        interceptor: Inteceptor[C, G, D],
        penalizer: Penalizer[C, G],
        sampler: Sampler[S],
        loader: Optional[Loader[S, D]] = None
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
        self._sampler = sampler
    @override
    def update(self) -> None:
        super().update()
        self._sampler.update()
    @override
    def sample[N: int](self, batch: N) -> Matrix[N, S]:
        return self._sampler.sample(batch)
    @override
    def restart(self) -> None:
        self._sampler.restart()
    @property
    @override
    def holdout(self) -> Matrix[int, S]:
        return self._sampler.holdout
