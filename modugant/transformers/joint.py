from typing import Any, Sequence

from modugant.conditioners.pooled import PooledConditioner
from modugant.interceptors.joint import JointInterceptor
from modugant.loaders.pooled import PooledLoader
from modugant.penalizers.joint import JointPenalizer
from modugant.protocols import Transformer
from modugant.transformers.composed import ComposedTransformer


class JointTransformer[S: int, C: int, G: int, D: int](ComposedTransformer[S, C, G, D]):
    '''Joint transformer for GANs.'''

    def __init__(
        self,
        conditions: C,
        intermediates: G,
        outputs: D,
        transformers: Sequence[Transformer[S, Any, Any, Any]]
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
            PooledConditioner(conditions, transformers),
            JointInterceptor(conditions, intermediates, outputs, transformers),
            JointPenalizer(conditions, intermediates, transformers),
            PooledLoader(outputs, transformers)
        )
