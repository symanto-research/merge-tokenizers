import ctypes
import glob
from collections import defaultdict
from ctypes import POINTER, c_int
from pathlib import Path

from ..types import Alignment, PositionAlignment, TokenAlignment, TokenizedPair
from ..utils.distances import get_distance_fn, precompute_distances
from .base import Aligner


class Tuple(ctypes.Structure):
    _fields_ = [("first", c_int), ("second", c_int)]


class AlignmentResult(ctypes.Structure):
    _fields_ = [("alignment", POINTER(Tuple)), ("n_elements", c_int)]


class DTWAligner(Aligner):
    def __init__(self, distance_name: str, radius: int = -1, **kwargs):
        super().__init__(**kwargs)
        self.distance_fn = get_distance_fn(distance_name)
        self.radius = radius
        self._build_c_lib()

    def _build_c_lib(self):
        """
        Loads the shared library and prepares the res and arg types.
        """
        so_library = glob.glob(f"{Path(__file__).parent}/dtw_c/*.so")[0]
        self.c_lib = ctypes.CDLL(so_library)
        self.c_lib.free_alignment_result.argtypes = [AlignmentResult]
        self.c_dtw = self.c_lib.dtw_alignment
        self.c_dtw.restype = AlignmentResult
        self.c_dtw.argtypes = [c_int, c_int, POINTER(c_int), c_int]

    def _align_pair(
        self,
        tokenized_pair: TokenizedPair,
    ) -> Alignment:
        """
        Aligns the tokens from two different tokenizers, using a
        C implementation of Dynamic Time Warping with radius.
        """
        # Add internal first token
        bos_tokens_a = ["<||BOS||>"] + tokenized_pair.preprocessed_tokens_a
        bos_tokens_b = ["<||BOS||>"] + tokenized_pair.preprocessed_tokens_b

        # Precompute distances
        distances = precompute_distances(
            bos_tokens_a, bos_tokens_b, self.distance_fn
        )

        # Compute alignments using c_dtw
        c_distances = (c_int * len(distances))(*distances)  # type: ignore
        alignment_result = self.c_dtw(
            len(bos_tokens_a), len(bos_tokens_b), c_distances, self.radius
        )
        alignments = [
            (
                alignment_result.alignment[i].first - 1,
                alignment_result.alignment[i].second - 1,
            )
            for i in range(alignment_result.n_elements)
        ][::-1][1:]

        # Free memory
        self.c_lib.free_alignment_result(alignment_result)

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
