from collections import defaultdict
from typing import List, Tuple

import numpy as np
from numba import njit

from ..types import Alignment, PositionAlignment, TokenAlignment, TokenizedPair
from ..utils.distances import get_distance_fn, precompute_distances
from .base import Aligner


@njit
def _dtw(len_a: int, len_b: int, distances: List[int], radius: int):
    """
    Computes Dynamic Time Warping and backtraces the pointers.
    """
    matrix = np.full(
        (len_a + 1, len_b + 1), np.iinfo(np.int32).max, dtype=np.int32
    )
    matrix[0][0] = 0
    # Compute DTW
    for i in range(1, len_a):
        for j in range(1, len_b):
            dist = distances[i * len_b + j]
            if radius > 0:
                if abs(i - j) <= radius:
                    matrix[i][j] = (
                        min(
                            matrix[i - 1][j],
                            matrix[i][j - 1],
                            matrix[i - 1][j - 1],
                        )
                        + dist
                    )
            else:
                matrix[i][j] = (
                    min(
                        matrix[i - 1][j], matrix[i][j - 1], matrix[i - 1][j - 1]
                    )
                    + dist
                )

    # Backtrace the pointers
    i, j = len_a, len_b
    alignment: List[Tuple[int, int]] = []
    while i > 0 and j > 0:
        min_ = min(matrix[i - 1][j], matrix[i][j - 1], matrix[i - 1][j - 1])
        if min_ == matrix[i - 1][j]:
            alignment.insert(0, (i - 1, j))
            i -= 1
        elif min_ == matrix[i][j - 1]:
            alignment.insert(0, (i, j - 1))
            j -= 1
        else:
            alignment.insert(0, (i - 1, j - 1))
            i -= 1
            j -= 1
    return alignment


class PythonDTWAligner(Aligner):
    def __init__(self, distance_name: str, radius: int = -1, **kwargs):
        super().__init__(**kwargs)
        self.distance_fn = get_distance_fn(distance_name)
        self.radius = radius

    def _align_pair(
        self,
        tokenized_pair: TokenizedPair,
    ) -> Alignment:
        """
        Aligns the tokens from two different tokenizers, using a
        Python implementation of Dynamic Time Warping with radius.
        """
        # Add internal first token
        bos_tokens_a = ["<||BOS||>"] + tokenized_pair.preprocessed_tokens_a
        bos_tokens_b = ["<||BOS||>"] + tokenized_pair.preprocessed_tokens_b

        # Precompute distances
        distances = precompute_distances(
            bos_tokens_a, bos_tokens_b, self.distance_fn
        )

        # Compute alignments
        alignments = _dtw(
            len(bos_tokens_a), len(bos_tokens_b), distances, self.radius
        )[1:]
        alignments = [(pos_a - 1, pos_b - 1) for pos_a, pos_b in alignments]

        # Merge alignments
        merged = defaultdict(list)

        for position_a, position_b in alignments:
            merged[position_a].append(position_b)

        # Conver to alignment types
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
