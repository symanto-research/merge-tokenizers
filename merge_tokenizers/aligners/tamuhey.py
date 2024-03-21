import spacy_alignments as tokenizations

from ..types import Alignment, PositionAlignment, TokenAlignment, TokenizedPair
from .base import Aligner


class TamuheyAligner(Aligner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _align_pair(
        self,
        tokenized_pair: TokenizedPair,
    ) -> Alignment:
        """
        Aligns the tokens from two different tokenizers, using
        the Tamuhey's algorithm: https://github.com/explosion/tokenizations
        """
        alignments, _ = tokenizations.get_alignments(
            tokenized_pair.preprocessed_tokens_a,
            tokenized_pair.preprocessed_tokens_b,
        )

        # Convert to alignment types
        position_alignments = [
            PositionAlignment(
                position_a=position_a, positions_b=alignments[position_a]
            )
            for position_a in range(len(alignments))
        ]
        token_alignments = [
            TokenAlignment(
                token_a=tokenized_pair.tokens_a[position_a],
                tokens_b=[
                    tokenized_pair.tokens_b[position_b]
                    for position_b in alignments[position_a]
                ],
            )
            for position_a in range(len(alignments))
        ]

        return Alignment(positions=position_alignments, tokens=token_alignments)
