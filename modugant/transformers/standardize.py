from typing import List

from torch import Tensor

from modugant.conditioners.none import NoneConditioner
from modugant.interceptors.direct import DirectInterceptor
from modugant.loaders.standardize import StandardizeLoader
from modugant.matrix.dim import Dim, Zero
from modugant.transformers.composed import ComposedTransformer


class StandardizeTransformer[D: int](ComposedTransformer[Zero, D, D]):
    '''Transformer for standardizing data.'''

    def __init__(
        self,
        dim: D,
        index: List[int],
        data: Tensor
    ):
        '''
        Initialize the transformer for standardizing data.

        Args:
            dim: The dimension of the data
            index: The indices of the data to standardize
            data: The data to standardize

        '''
        super().__init__(
            conditioner = NoneConditioner(dim),
            interceptor = DirectInterceptor(Dim.zero(), dim),
            loader = StandardizeLoader(
                dim,
                data,
                index
            )
        )
