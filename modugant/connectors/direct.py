from typing import Optional

from modugant.conditioners import NoneConditioner
from modugant.connectors.composed import ComposedConnector
from modugant.interceptors import DirectInterceptor
from modugant.matrix.dim import Dim, Zero
from modugant.penalizers import StaticPenalizer
from modugant.protocols import Loader, Penalizer, Sampler


class DirectConnector[D: int](ComposedConnector[Zero, D, D]):
    '''Direct connector for GANs.'''

    def __init__(
        self,
        dim: D,
        sampler: Sampler[int],
        penalizer: Optional[Penalizer[Zero, D]] = None,
        loader: Optional[Loader[D]] = None
    ) -> None:
        '''
        Initialize the direct connector.

        Args:
            dim (int): The number of inputs and outputs.
            sampler (Sampler): The sampler.
            penalizer (Penalizer): The penalizer.
            loader (Loader): The loader.

        '''
        super().__init__(
            NoneConditioner(dim),
            DirectInterceptor(Dim.zero(), dim),
            penalizer or StaticPenalizer(Dim.zero(), dim),
            sampler,
            loader
        )
