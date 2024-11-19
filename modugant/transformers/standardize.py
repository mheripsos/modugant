from torch import Tensor

from modugant.conditioners.none import NoneConditioner
from modugant.interceptors.direct import DirectInterceptor
from modugant.loaders.standardize import StandardizeLoader
from modugant.matrix.dim import Dim, Zero
from modugant.matrix.index import Index
from modugant.transformers.composed import ComposedTransformer


class StandardizeTransformer[S: int, D: int](ComposedTransformer[S, Zero, D, D]):
    '''Transformer for standardizing data.'''

    def __init__(
        self,
        sampled: S,
        dim: D,
        index: Index[D, S],
        data: Tensor
    ):
        '''
        Initialize the transformer for standardizing data.

        Args:
            sampled: The number of sampled dimensions
            dim: The dimension of the data
            index: The indices of the data to standardize
            data: The data to standardize

        '''
        super().__init__(
            conditioner = NoneConditioner(sampled),
            interceptor = DirectInterceptor(Dim.zero(), dim),
            loader = StandardizeLoader(data, index)
        )
