from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Union

import numpy as np

from ..types import Alignment, TokenizedPair, TokenizedSet
from ..utils.heuristics import align_one_to_one
from ..utils.preprocess import preprocess_tokens


class Aligner(ABC):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @abstractmethod
    def _align_pair(self, tokenized_pair: TokenizedPair) -> Alignment:
        """
        Aligns the tokens from two different tokenizers.

        Args:
            tokenized_pair (TokenizedPair): a pair of tokenized texts.

        Returns:
            Alignment: positions and tokens of the alignment.
        """
        ...

    def align_pair(
        self,
        tokenized_pair: TokenizedPair,
    ) -> Alignment:
        """
        Preprocess the tokens of a tokenized pair and aligns them.

        Args:
            tokenized_pair (TokenizedPair): a pair of tokenized texts.

        Returns:
            Alignment: positions and tokens of the alignment.
        """
        tokenized_pair.preprocessed_tokens_a = preprocess_tokens(
            tokenized_pair.tokens_a
        )
        tokenized_pair.preprocessed_tokens_b = preprocess_tokens(
            tokenized_pair.tokens_b
        )

        # If both tokenizations are the same, return 1-1 alignment
        if (
            tokenized_pair.preprocessed_tokens_a
            == tokenized_pair.preprocessed_tokens_b
        ):
            return align_one_to_one(tokenized_pair)

        return self._align_pair(tokenized_pair)

    def align(self, tokenized_set: TokenizedSet) -> List[Alignment]:
        """
        Aligns the tokens from multiple tokenizers, picking the first
        tokenizer as reference and align the other ones with it.

        Args:
            tokenized_set (TokenizedSet): multiple tokenized texts.

        Returns:
            List[Alignment]: positions and tokens of each alignment.
        """
        word_ids = (
            tokenized_set.word_ids
            if tokenized_set.word_ids
            else [[] for _ in range(len(tokenized_set.tokens))]
        )
        spans = (
            tokenized_set.spans
            if tokenized_set.spans
            else [[] for _ in range(len(tokenized_set.tokens))]
        )

        tokens_a = tokenized_set.tokens[0]
        word_ids_a = word_ids[0]
        spans_a = spans[0]

        return [
            self.align_pair(
                TokenizedPair(
                    tokens_a=tokens_a,
                    tokens_b=tokens_b,
                    word_ids_a=word_ids_a,
                    word_ids_b=word_ids_b,
                    spans_a=spans_a,
                    spans_b=spans_b,
                    text=tokenized_set.text,
                )
            )
            for tokens_b, word_ids_b, spans_b in zip(
                tokenized_set.tokens[1:], word_ids[1:], spans[1:]
            )
        ]

    def aggregate_features_pair(
        self,
        tokenized_pair: TokenizedPair,
        aggregate_fn: Callable = np.mean,
        alignment: Optional[Alignment] = None,
    ) -> np.ndarray:
        """
        Aggregates features associated to the tokens after aligning
        the tokens of two tokenizers.

        Args:
            tokenized_pair (TokenizedPair): a pair of tokenized texts.
            alignment (Alignment): positions and tokens of the alignment.

        Returns:
            np.ndarray: features of `tokens_b` aggregated to match the tokens of `tokens_a`.
        """
        assert (
            tokenized_pair.features_a is not None
            and tokenized_pair.features_b is not None
        ), "Features can't be None"

        if alignment is None:
            alignment = self.align_pair(tokenized_pair)

        aggregated_features = np.zeros(
            (
                tokenized_pair.features_a.shape[0],
                *tokenized_pair.features_b.shape[1:],
            )
        )

        for position_a, positions_b in alignment:
            aggregated_features[position_a] = aggregate_fn(
                tokenized_pair.features_b[positions_b]
            )

        return aggregated_features

    def aggregate_features(
        self,
        tokenized_set: TokenizedSet,
        aggregate_fn: Callable = np.mean,
        stack: bool = False,
        alignments: Optional[List[Alignment]] = None,
    ) -> Union[List[np.ndarray], np.ndarray]:
        """
        Aggregates features associated to the tokens after aligning
        the tokens of multiple tokenizers.

        The alignment of multiple tokenizers is done by picking the first
        tokenizer as reference and aligning the other ones with it.

        Args:
            tokenized_set (TokenizedSet): multiple tokenized texts.
            aggregate_fn (Callable): function to aggregate the tokens of `tokens_b` matched with
                                     each token in `tokens_a`.
            stack (bool): whether to stack horizontally all the features after aligning the tokens.
            alignments (List[Alignment]): positions and tokens of each alignment.

        Returns:
            Union[List[np.ndarray], np.ndarray]: np.ndarray with the stacked features if `stack`
                                                 is True, else, a list of the features.
        """
        assert (
            len(tokenized_set.features) > 0
        ), "Features must be a non-empty list of numpy arrays."

        assert len(tokenized_set.features) == len(
            tokenized_set.tokens
        ), "There are less features than tokenizations."

        word_ids = (
            tokenized_set.word_ids
            if tokenized_set.word_ids
            else [[] for _ in range(len(tokenized_set.tokens))]
        )

        spans = (
            tokenized_set.spans
            if tokenized_set.spans
            else [[] for _ in range(len(tokenized_set.tokens))]
        )

        tokens_a = tokenized_set.tokens[0]
        word_ids_a = word_ids[0]
        spans_a = spans[0]
        features_a = tokenized_set.features[0]
        merged_features = []

        for idx, (tokens_b, features_b, word_ids_b, spans_b) in enumerate(
            zip(
                tokenized_set.tokens[1:],
                tokenized_set.features[1:],
                word_ids[1:],
                spans[1:],
            )
        ):
            merged_features.append(
                self.aggregate_features_pair(
                    TokenizedPair(
                        tokens_a=tokens_a,
                        tokens_b=tokens_b,
                        word_ids_a=word_ids_a,
                        word_ids_b=word_ids_b,
                        spans_a=spans_a,
                        spans_b=spans_b,
                        features_a=features_a,
                        features_b=features_b,
                        text=tokenized_set.text,
                    ),
                    aggregate_fn,
                    alignment=(
                        alignments[idx] if alignments is not None else None
                    ),
                )
            )

        if stack:
            return np.hstack((tokenized_set.features[0], *merged_features))

        return [tokenized_set.features[0], *merged_features]
