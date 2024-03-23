from ..types import Alignment, PositionAlignment, TokenAlignment, TokenizedPair
from ..utils.distances import get_distance_fn
from .base import Aligner


class GreedyDistanceAligner(Aligner):
    def __init__(self, distance_name: str, radius: int = 30, **kwargs):
        super().__init__(**kwargs)
        self.distance_fn = get_distance_fn(distance_name)
        assert radius > 0, "Radius must be greater than 0."
        self.radius = radius

    def _align_pair(
        self,
        tokenized_pair: TokenizedPair,
    ) -> Alignment:
        """
        Aligns the tokens from two different tokenizers, using
        a greedy matching algorithm based on token distances.

        Matches each token with the token in a surrounding window
        according to the following eq if `word_ids` are passed:

        match(t_i) = min_{t_j} dist(t_i, t_j) * s_ij
        s_ij = {1 if (word_id(t_i) == word_id(t_j)), inf otherwise}

        or the following eq if `word_ids` are not passed:

        match(t_i) = min_{t_j} dist(t_i, t_j)

        """
        alignments = []
        len_a = len(tokenized_pair.preprocessed_tokens_a)
        len_b = len(tokenized_pair.preprocessed_tokens_b)
        for i in range(len_a):
            start = max(0, i - self.radius)
            end = min(len_b, i + self.radius)
            min_dist = float("inf")
            match_position = -1
            # If len_a > len_b, add all the remaining b tokens to the last of a
            if start >= end:
                alignments.append((i, [len_b - 1]))
            else:
                for j in range(start, end):
                    dist = self.distance_fn(
                        tokenized_pair.preprocessed_tokens_a[i],
                        tokenized_pair.preprocessed_tokens_b[j],
                    )
                    if tokenized_pair.word_ids_a and tokenized_pair.word_ids_b:
                        s = (
                            float("inf")
                            if tokenized_pair.word_ids_a[i]
                            != tokenized_pair.word_ids_b[j]
                            else 1
                        )
                        dist *= s
                    if dist < min_dist:
                        min_dist, match_position = dist, j
                alignments.append((i, [match_position]))

        # Convert to alignment types
        position_alignments = [
            PositionAlignment(position_a=position_a, positions_b=positions_b)
            for position_a, positions_b in alignments
        ]
        token_alignments = [
            TokenAlignment(
                token_a=tokenized_pair.tokens_a[position_a],
                tokens_b=[
                    tokenized_pair.tokens_b[position_b]
                    for position_b in positions_b
                ],
            )
            for position_a, positions_b in alignments
        ]

        return Alignment(positions=position_alignments, tokens=token_alignments)
