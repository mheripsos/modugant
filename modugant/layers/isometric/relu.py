from torch.nn import LeakyReLU

from modugant.layers.isometric.isometric import IsometricLayer


class RectifiedLayer[D: int](IsometricLayer[D], LeakyReLU):
    '''Rectified linear unit layer.'''

    def __init__(self, dim: D, slope: float = 0.0) -> None:
        '''
        Initialize the rectified linear unit layer.

        Args:
            dim (int): The number of nodes.
            slope (float): The slope of the negative part.

        '''
        super().__init__(slope)
        self._dim = dim
