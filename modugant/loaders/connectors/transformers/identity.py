from typing import Any, override

from modugant.loaders.connectors.transformers.protocol import Transformer
from modugant.matrix.index import Index
from modugant.matrix.matrix import Matrix


class IdentityTransformer[S: int](Transformer[S]):
    '''Identity Transformer for Connector composition. Simply passes through the raw data.'''

    def __init__(self, index: Index[S, int]) -> None:
        '''
        Initialize the identity transformer.

        Args:
            index (Index): The index to load.

        '''
        self._index = index
    @override
    def load[N: int](self, data: Matrix[N, Any]) -> Matrix[N, S]:
        '''Transform underlying data.'''
        return data[..., self._index]
    @override
    def unload[N: int](self, data: Matrix[N, S]) -> Matrix[N, Any]:
        '''Revert to underlying data.'''
        return data[..., ...]
