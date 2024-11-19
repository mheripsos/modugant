from typing import Any, Sequence, override

from modugant.matrix import Matrix
from modugant.protocols import Connector, Sampler, Transformer
from modugant.transformers import JointTransformer


class JointConnector[S: int, C: int, G: int, D: int](JointTransformer[S, C, G, D], Connector[S, C, G, D]):
    '''Joint connector for GANs.'''

    def __init__(
        self,
        conditions: C,
        intermediates: G,
        outputs: D,
        transformers: Sequence[Transformer[S, Any, Any, Any]],
        sampler: Sampler[S]
    ) -> None:
        '''
        Initialize the joint connector.

        Args:
            conditions (int): The number of conditions.
            intermediates (int): The number of generated outputs.
            outputs (int): The number of transformed outputs.
            transformers (Sequence[Transformer]): The transformers.
            sampler (Sampler): The sampler

        '''
        assert sum([transformer.conditions for transformer in transformers]) == conditions
        assert sum([transformer.intermediates for transformer in transformers]) == intermediates
        assert sum([transformer.outputs for transformer in transformers]) == outputs
        super().__init__(conditions, intermediates, outputs, transformers)
        self._sampler = sampler
    @override
    def update(self) -> None:
        super().update()
        self._sampler.update()
    @override
    def sample[N: int](self, batch: N) -> Matrix[N, S]:
        return self._sampler.sample(batch)
    @override
    def restart(self) -> None:
        self._sampler.restart()
    @property
    @override
    def holdout(self) -> Matrix[int, S]:
        return self._sampler.holdout
