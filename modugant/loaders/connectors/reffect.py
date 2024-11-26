
from typing import Tuple, override

from modugant.loaders.connectors.protocol import Connector
from modugant.loaders.connectors.splitters.identity import IdentitySplitter
from modugant.loaders.connectors.transformers.onehot import OneHotTransformer
from modugant.loaders.connectors.transformers.reffect import EffectGroup, RandomEffectTransformer
from modugant.matrix.dim import One, Zero
from modugant.matrix.index import Index
from modugant.matrix.matrix import Matrix


class RandomEffectConnector[U: int, G: int, S: int](RandomEffectTransformer[U, G, S], Connector[S, S, Zero]):
    '''
    Connector for Loader composition.

    Type Parameters:
        U: The dimensionality of the random effect matrix
        G: The dimensionality of the group matrix
        D: The dimensionality of the discriminable data

    '''

    def __init__(
        self,
        dim: S,
        index: Tuple[int, U],
        groups: EffectGroup[G, U],
        lr: float = 0.001
    ) -> None:
        '''
        Initialize the random effect connector.

        Args:
            dim (int): The number of outputs.
            index (Tuple[int, U]): The index of the data to load.
            groups (Tuple[Index, List[EffectGroup]]): The grouping index.
            lr (float): The learning rate

        '''
        super().__init__(
            OneHotTransformer(index),
            dim,
            groups,
            lr
        )
        self._samples = dim
        self._groups = groups
        self._lr = lr
        self._splitter = IdentitySplitter(Index.empty(dim), Index.range(dim))
    @override
    def condition[N: int](self, data: Matrix[N, S]) -> Matrix[N, S]:
        return self._splitter.condition(data)
    @override
    def prepare[N: int](self, data: Matrix[N, S]) -> Matrix[N, Zero]:
        return self._splitter.prepare(data)
    @override
    def intercept[N: int](self, condition: Matrix[N, S], intermediate: Matrix[N, Zero]) -> Matrix[N, Zero]:
        return intermediate
    @override
    def loss[N: int](self, condition: Matrix[N, S], intermediate: Matrix[N, Zero]) -> Matrix[One, One]:
        return self.penalty()
