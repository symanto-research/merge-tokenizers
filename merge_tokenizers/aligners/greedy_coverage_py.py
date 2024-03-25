from collections import defaultdict
from typing import List, Tuple

from ..types import Alignment, PositionAlignment, TokenAlignment, TokenizedPair
from .base import Aligner


def get_spans(tokens: List[str], text: str) -> List[Tuple[int, int]]:
    """
    Finds the positions (start, end) that each token in
    `tokens` covers in the text.

    Args:
        tokens (List[str]): list of tokens
        text (str): text without whitespaces
    Returns:
        List[Tuple[int, int]]: (start, end) positions for each token
    """
    spans = []
    j = 0
    for token in tokens:
        start_pos = None
        end_pos = None
        matches = 0
        for i in range(len(token)):
            if token[i] == text[j + matches]:
                if start_pos is None:
                    start_pos = matches + j
                end_pos = matches + j + 1
                matches += 1

        if end_pos is not None:
            j = end_pos

        if start_pos is not None and end_pos is not None:
            spans.append((start_pos, end_pos))
            if j >= len(text):
                break
        else:
            spans.append((-1, -1))

    # Add missing last span in case of </s>
    while len(spans) < len(tokens):
        spans.append((-1, -1))
    return spans


def merge_spans(
    spans_a: List[Tuple[int, int]], spans_b: List[Tuple[int, int]]
) -> List[Tuple[int, int]]:
    """
    Merge the tokens of `b` that are spanned by the tokens
    of `a` according to their (start, end) positions.

    Example:
        spans_a = [(0, 5), (5, 13), (13, 23)]
        spans_b = [(0, 4), (5, 8), (8, 11), (11, 14), (15, 19), (19, 21), (21, 22)]
        result = [(0, [0]), (1, [1, 2, 3]), (2, [4, 5, 6])]

    Args:
        spans_a (List[Tuple[int, int]]): (start, end) positions of the tokens in one tokenization
        spans_b (List[Tuple[int, int]]): (start, end) positions of the tokens in another tokenization

    Returns:
        List[Tuple[int, int]]:
    """
    alignments = []
    i, j = 0, 0
    while i < len(spans_a) and j < len(spans_b):
        a_start, a_end = spans_a[i]
        b_start, b_end = spans_b[j]
        alignments.append((i, j))

        if a_start == b_start and a_end == b_end:
            i += 1
            j += 1
        elif a_end == b_end:
            i += 1
            j += 1
        elif a_end <= b_end:
            i += 1
        else:
            j += 1

    # Add remaining alignments
    while i < len(spans_a):
        alignments.append((i, j - 1))
        i += 1

    while j < len(spans_b):
        alignments.append((i - 1, j))
        j += 1
    return alignments


class PythonGreedyCoverageAligner(Aligner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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

        # Get spans
        # If the spans covering the text are not passed, compute them.
        if not tokenized_pair.spans_a and not tokenized_pair.spans_b:
            assert (
                tokenized_pair.text
            ), "`text` must be passed as argument when not passing `span_a` and `span_b`"
            text = tokenized_pair.text.lower().replace(" ", "")
            spans_a = get_spans(tokenized_pair.preprocessed_tokens_a, text)
            spans_b = get_spans(tokenized_pair.preprocessed_tokens_b, text)
        # Otherwise, use them.
        else:
            spans_a = tokenized_pair.spans_a
            spans_b = tokenized_pair.spans_b

        # Align spans
        alignments = merge_spans(spans_a, spans_b)

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
