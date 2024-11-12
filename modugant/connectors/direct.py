from typing import Optional

from modugant.conditioners import NoneConditioner
from modugant.connectors.composed import ComposedConnector
from modugant.interceptors import DirectInterceptor
from modugant.matrix.dim import Dim, Zero
from modugant.protocols import Loader, Sampler, Updater
from modugant.updaters import StaticUpdater


class DirectConnector[D: int](ComposedConnector[Zero, D, D]):
    '''Direct connector for GANs.'''

    def __init__(
        self,
        dim: D,
        sampler: Sampler[int],
        updater: Optional[Updater[Zero, D]] = None,
        loader: Optional[Loader[D]] = None
    ) -> None:
        '''
        Initialize the direct connector.

        Args:
            dim (int): The number of inputs and outputs.
            sampler (Sampler): The sampler.
            updater (Updater): The updater.
            loader (Loader): The loader.

        '''
        super().__init__(
            NoneConditioner(dim),
            DirectInterceptor(Dim.zero(), dim),
            updater or StaticUpdater(Dim.zero(), dim),
            sampler,
            loader
        )
