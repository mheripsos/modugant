from typing import List, Tuple

from modugant.conditioners.block import BlockConditioner
from modugant.interceptors.softmax import SoftmaxInterceptor
from modugant.loaders.onehot import OneHotLoader
from modugant.penalizers.entropy import EntropyPenalizer
from modugant.transformers.composed import ComposedTransformer


class CategoryTransformer[S: int, B: int](ComposedTransformer[S, B, B, B]):
    '''Transformer for a single one-hot category.'''

    def __init__(
        self,
        sampled: S,
        index: Tuple[int, B]
    ) -> None:
        '''
        Initialize the transformer for a single category.

        Args:
            sampled: The number of sampled dimensions
            index: The start index in the original data and the size (bins) of the category

        '''
        (_, bins) = index
        super().__init__(
            conditioner = BlockConditioner(
                sampled,
                bins,
                index = [(0, bins)],
                picks = 1
            ),
            interceptor = SoftmaxInterceptor(bins, bins, bins, index = [(0, bins)]),
            penalizer = EntropyPenalizer(bins, bins, index = [(0, 0, bins)]),
            loader = OneHotLoader(
                sampled,
                bins,
                index = [index]
            )
        )

class CategoriesTransformer[S: int, B: int](ComposedTransformer[S, B, B, B]):
    '''Transformer for many one-hot categories.'''

    def __init__(
        self,
        sampled: S,
        width: B,
        index: List[Tuple[int, int]],
        picks: int = 1
    ) -> None:
        '''
        Initialize the transformer for many categories.

        Args:
            sampled: The number of sampled dimensions
            width: The total size of the category
            index: A list of tuples of the start index in the original data and the size (bins) of the category
            picks: The number of category blocks to sample

        '''
        sizes = [size for (_, size) in index]
        assert sum(sizes) == width
        ## The index and size of categories after being loaded
        cumu = [(sum(sizes[:i]), sizes[i]) for i in range(len(index))]
        super().__init__(
            conditioner = BlockConditioner(
                sampled,
                width,
                index = cumu,
                picks = picks
            ),
            interceptor = SoftmaxInterceptor(
                width,
                width,
                width,
                index = cumu
            ),
            penalizer = EntropyPenalizer(
                width,
                width,
                index = [(start, start, size) for (start, size) in cumu]
            ),
            loader = OneHotLoader(
                sampled,
                width,
                index = index
            )
        )
