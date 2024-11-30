from torch.nn import BatchNorm1d

from modugant.layers.isometric.isometric import IsometricLayer


class BatchNormLayer[D: int](IsometricLayer[D], BatchNorm1d):
    '''Batch normalization layer.'''

    def __init__(self, dim: D) -> None:
        '''
        Initialize the batch normalization layer.

        Args:
            dim (int): The number of nodes.

        '''
        super().__init__(dim)
        self._dim = dim
