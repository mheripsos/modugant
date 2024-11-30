from torch.nn import Sigmoid

from modugant.layers.isometric.isometric import IsometricLayer


class SigmoidLayer[D: int](IsometricLayer[D], Sigmoid):
    '''Sigmoid layer.'''

    def __init__(self, dim: D) -> None:
        '''
        Initialize the sigmoid layer.

        Args:
            dim (int): The number of nodes.

        '''
        super().__init__()
        self._dim = dim
