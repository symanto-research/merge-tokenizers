from collections import defaultdict

from fastdtw import fastdtw
from sklearn.feature_extraction.text import CountVectorizer

from ..types import Alignment, PositionAlignment, TokenAlignment, TokenizedPair
from ..utils.distances import get_distance_fn
from .base import Aligner


class FastDTWAligner(Aligner):
    def __init__(self, distance_name: str, radius: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.distance_fn = get_distance_fn(distance_name)
        self.radius = radius

    def _align_pair(
        self,
        tokenized_pair: TokenizedPair,
    ) -> Alignment:
        """
        Aligns the tokens from two different tokenizers, using
        FastDTW and bag of characters to represent tokens.
        """

        vectorizer = CountVectorizer(analyzer="char").fit(
            tokenized_pair.preprocessed_tokens_a
            + tokenized_pair.preprocessed_tokens_b
        )
        boc_tokens_a = vectorizer.transform(
            tokenized_pair.preprocessed_tokens_a
        ).toarray()
        boc_tokens_b = vectorizer.transform(
            tokenized_pair.preprocessed_tokens_b
        ).toarray()

        # Compute alignments using fastdtw
        _, alignments = fastdtw(boc_tokens_a, boc_tokens_b, radius=self.radius)

        # Merge alignments
        merged = defaultdict(list)

        for position_a, position_b in alignments:
            merged[position_a].append(position_b)

        # Convert to alignment types
        position_alignments = [
            PositionAlignment(position_a=position_a, positions_b=positions_b)
            for position_a, positions_b in merged.items()
        ]
        token_alignments = [
            TokenAlignment(
                token_a=tokenized_pair.tokens_a[position_a],
                tokens_b=[
                    tokenized_pair.tokens_b[position_b]
                    for position_b in positions_b
                ],
            )
            for position_a, positions_b in merged.items()
        ]

        return Alignment(positions=position_alignments, tokens=token_alignments)
