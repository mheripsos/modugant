from torch.nn import Dropout

from modugant.layers.isometric.isometric import IsometricLayer


class DropoutLayer[D: int](IsometricLayer[D], Dropout):
    '''Dropout layer.'''

    def __init__(self, dim: D, rate: float = 0.5) -> None:
        '''
        Initialize the dropout layer.

        Args:
            dim (int): The number of output nodes.
            rate (float): The dropout rate.

        '''
        super().__init__(rate)
        self._dim = dim
