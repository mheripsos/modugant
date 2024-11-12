from typing import List, Tuple, override

from modugant.matrix import Matrix
from modugant.matrix.dim import One
from modugant.matrix.index import Index
from modugant.matrix.ops import cross_entropy, sums
from modugant.protocols import Updater


class EntropyUpdater[C: int, G: int](Updater[C, G]):
    '''Entropy updater for GANs.'''

    def __init__(
        self,
        conditions: C,
        intermediates: G,
        index: List[Tuple[int, int, int]]
    ) -> None:
        '''
        Initialize the entropy updater.

        Args:
            conditions (C: int): The number of conditions.
            intermediates (G: int): The number of generated outputs.
            index (List[Tuple[int, int, int]]): The start indices of the condition and output, and the size
                e.g. [(0, 3, 3)] defines a single block where condition[:, 0:3] and output[:, 3:6] are compared.

        '''
        assert sum([size for (_, _, size) in index]) == conditions
        self._conditions = conditions
        self._intermediates = intermediates
        self._index = [
            (Index.slice(c_start, block), Index.slice(d_start, block))
            for (c_start, d_start, block) in index
        ]
    @override
    def loss[N: int](self, condition: Matrix[N, C], intermediate: Matrix[N, G]) -> Matrix[One, One]:
        losses = tuple(
            (
                condition[..., c_idx].sum(dim = 1, keepdim = True) *
                cross_entropy(
                    intermediate[..., d_idx],
                    condition[..., c_idx].argmax(dim = 1, keepdim = True),
                    reduction = 'none'
                )
            )
            for (c_idx, d_idx) in self._index
        )
        return sums(losses).sum(dim = 0, keepdim = True) / condition.shape[0]
    @override
    def update(self) -> None:
        pass
