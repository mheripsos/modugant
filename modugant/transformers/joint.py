from typing import Any, Sequence

from modugant.conditioners.joint import JointConditioner
from modugant.interceptors.joint import JointInterceptor
from modugant.loaders.joint import JointLoader
from modugant.protocols import Transformer
from modugant.transformers.composed import ComposedTransformer
from modugant.updaters.joint import JointUpdater


class JointTransformer[C: int, G: int, D: int](ComposedTransformer[C, G, D]):
    '''Joint transformer for GANs.'''

    def __init__(
        self,
        conditions: C,
        intermediates: G,
        outputs: D,
        transformers: Sequence[Transformer[Any, Any, Any]]
    ) -> None:
        '''
        Initialize the joint transformer.

        Args:
            conditions (C: int): The number of conditions.
            intermediates (G: int): The number of generated outputs.
            outputs (D: int): The number of transformed outputs.
            transformers (Sequence[Transformer]): The transformers.

        '''
        assert sum([transformer.conditions for transformer in transformers]) == conditions
        assert sum([transformer.intermediates for transformer in transformers]) == intermediates
        assert sum([transformer.outputs for transformer in transformers]) == outputs
        super().__init__(
            JointConditioner(conditions, outputs, transformers),
            JointInterceptor(conditions, intermediates, outputs, transformers),
            JointUpdater(conditions, intermediates, transformers),
            JointLoader(outputs, transformers)
        )
