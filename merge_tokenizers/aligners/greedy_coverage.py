import ctypes
import glob
from collections import defaultdict
from ctypes import POINTER, c_char_p, c_int
from pathlib import Path

from ..types import Alignment, PositionAlignment, TokenAlignment, TokenizedPair
from .base import Aligner


class Tuple(ctypes.Structure):
    _fields_ = [("start", c_int), ("end", c_int)]


class AlignmentResult(ctypes.Structure):
    _fields_ = [("alignment", POINTER(Tuple)), ("n_elements", c_int)]


class GreedyCoverageAligner(Aligner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._build_c_lib()

    def _build_c_lib(self):
        """
        Loads the shared library and prepares the res and arg types.
        """
        so_library = glob.glob(
            f"{Path(__file__).parent}/greedy_coverage_c/*.so"
        )[0]
        self.c_lib = ctypes.CDLL(so_library)

        self.c_get_spans = self.c_lib.get_spans
        self.c_merge_spans = self.c_lib.merge_spans
        self.c_free_spans = self.c_lib.free_spans
        self.c_free_alignment = self.c_lib.free_alignment

        self.c_get_spans.restype = POINTER(Tuple)
        self.c_get_spans.argtypes = [
            POINTER(c_char_p),
            c_char_p,
            c_int,
        ]

        self.c_merge_spans.restype = AlignmentResult
        self.c_merge_spans.argtypes = [
            POINTER(Tuple),
            POINTER(Tuple),
            c_int,
            c_int,
        ]

        self.c_free_spans.argtypes = [POINTER(Tuple)]
        self.c_free_alignment.argtypes = [AlignmentResult]

    def _align_pair(
        self,
        tokenized_pair: TokenizedPair,
    ) -> Alignment:
        """
        Aligns the tokens from two different tokenizers, using
        a greedy matching algorithm based on text coverage.

        The procedure remove whitespaces from the text, and
        finds the positions (start, end) that each token covers
        in the text without whitespaces. Once we have the lists of (start, end)
        for each token and for each tokenization, we merge the tokens of the
        second tokenization that are spanned by the tokens of the first tokenization.

        For instance:
        spans_a = [(0, 5), (5, 13), (13, 23)]
        spans_b = [(0, 4), (5, 8), (8, 11), (11, 14), (15, 19), (19, 21), (21, 23)]

        will result in [(0, [0]), (1, [1, 2, 3]), (2, [4, 5, 6])]
        """

        # Get the span covered by each token
        spans = {}
        # If the spans covering the text are not passed, compute them.
        if not tokenized_pair.spans_a and not tokenized_pair.spans_b:
            text = tokenized_pair.text.lower().replace(" ", "").encode("utf-8")
            for tokenization, preprocessed_tokens in {
                "a": tokenized_pair.preprocessed_tokens_a,
                "b": tokenized_pair.preprocessed_tokens_b,
            }.items():
                ptr = (ctypes.c_char_p * len(preprocessed_tokens))(
                    *[token.encode("utf-8") for token in preprocessed_tokens]
                )
                c_spans = self.c_get_spans(
                    ptr,
                    text,
                    len(preprocessed_tokens),
                )
                spans[tokenization] = [
                    (c_spans[i].start, c_spans[i].end)
                    for i in range(len(preprocessed_tokens))
                ]
                self.c_free_spans(c_spans)
        # Otherwise, use them
        else:
            spans["a"] = tokenized_pair.spans_a
            spans["b"] = tokenized_pair.spans_b

        # Merge the spans
        c_spans_a = (Tuple * len(spans["a"]))(*spans["a"])  # type: ignore
        c_spans_b = (Tuple * len(spans["b"]))(*spans["b"])  # type: ignore
        c_alignments = self.c_merge_spans(
            c_spans_a, c_spans_b, len(spans["a"]), len(spans["b"])
        )
        alignments = [
            (c_alignments.alignment[i].start, c_alignments.alignment[i].end)
            for i in range(c_alignments.n_elements)
        ]
        self.c_free_alignment(c_alignments)

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
