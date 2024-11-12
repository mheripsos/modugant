from typing import List, Tuple

from modugant.conditioners.category import CategoryConditioner
from modugant.interceptors.softmax import SoftmaxInterceptor
from modugant.loaders.category import CategoryLoader
from modugant.transformers.composed import ComposedTransformer
from modugant.updaters.entropy import EntropyUpdater


class CategoryTransformer[B: int](ComposedTransformer[B, B, B]):
    '''Transformer for a single category.'''

    def __init__(
        self,
        index: Tuple[int, B] # no longer fixed to (4, 3)
    ) -> None:
        '''
        Initialize the transformer for a single category.

        Args:
            index: The start index in the original data and the size (bins) of the category

        '''
        (_, bins) = index
        super().__init__(
            conditioner = CategoryConditioner(
                bins,
                bins,
                index = [(0, bins)],
                samples = 1
            ),
            interceptor = SoftmaxInterceptor(bins, bins, bins, index = [(0, bins)]),
            updater = EntropyUpdater(bins, bins, index = [(0, 0, bins)]),
            loader = CategoryLoader(
                bins,
                index = [index]
            )
        )

class CategoriesTransformer[B: int](ComposedTransformer[B, B, B]):
    '''Transformer for many categorioes.'''

    def __init__(
        self,
        width: B, # total size now needs to be specified, and we will assert it
        index: List[Tuple[int, int]], # take in a list of indices instead
        samples: int = 1
    ) -> None:
        '''
        Initialize the transformer for many categories.

        Args:
            width: The total size of the category
            index: A list of tuples of the start index in the original data and the size (bins) of the category
            samples: The number of category blocks to sample

        '''
        sizes = [size for (_, size) in index]
        assert sum(sizes) == width
        ## The index and size of categories after being loaded
        cumu = [(sum(sizes[:i]), sizes[i]) for i in range(len(index))]
        super().__init__(
            conditioner = CategoryConditioner(
                width,
                width,
                index = cumu,
                samples = samples
            ),
            interceptor = SoftmaxInterceptor(
                width,
                width,
                width,
                index = cumu
            ),
            updater = EntropyUpdater(
                width,
                width,
                # category index and data index are the same
                index = [(start, start, size) for (start, size) in cumu]
            ),
            loader = CategoryLoader(
                width,
                index = index
            )
        )
