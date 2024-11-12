from typing import Callable

from torch.nn import Module, Sequential

from modugant.discriminators.basic import BasicDiscriminator


class StandardDiscriminator[C: int, D: int](BasicDiscriminator[C, D]):
    '''
    Standard discriminator model for GANs.

    Type Parameters:
        C: The number of condition nodes.
        D: The dimensionality of the data.

    '''

    def __init__(
        self,
        conditions: C,
        outputs: D,
        steps: list[int],
        layer: Callable[[int, int, int], Module],
        finish: Callable[[int], Module]
    ) -> None:
        '''
        Initialize the discriminator model.

        Args:
            conditions (C: int): The number of condition nodes.
            outputs (D: int): The number of input nodes.
            steps (list[int]): The number of nodes in each step of the discriminator.
            layer (Callable[[int, int, int], nn.Module]): The step layer constructor.
                (inputs, outputs, index) -> layer
            finish (Callable[[int], nn.Module]): The final layer constructor.
                (inputs) -> layer
            optimizer (Callable[[], Optimizer]): The optimizer constructor.

        '''
        super().__init__(conditions, outputs)
        self._model = Sequential(
            *[
                layer(
                    steps[i - 1] if i else outputs + conditions,
                    steps[i],
                    i
                )
                for i in range(len(steps))
            ],
            finish(steps[-1] if len(steps) else outputs + conditions)
        )
