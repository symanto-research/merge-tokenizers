from typing import List

import numpy as np
from pydantic import BaseModel, field_validator, model_validator


class TokenizedPair(BaseModel):
    """
    Pair of tokenized texts.
    """

    tokens_a: List[str]
    tokens_b: List[str]
    word_ids_a: List[int] = []
    word_ids_b: List[int] = []
    preprocessed_tokens_a: List[str] = []
    preprocessed_tokens_b: List[str] = []
    features_a: np.ndarray = None
    features_b: np.ndarray = None

    @field_validator("word_ids_a", "word_ids_b", mode="before")
    @classmethod
    def prepare_word_ids_a(cls, word_ids):
        if word_ids:
            if word_ids[0] is None:
                word_ids[0] = -1
            if word_ids[-1] is None:
                word_ids[-1] = max(word_ids[:-1]) + 1
            return word_ids
        else:
            return []

    class Config:
        arbitrary_types_allowed = True


class TokenizedSet(BaseModel):
    """
    Multiple tokenized texts.
    """

    tokens: List[List[str]]
    word_ids: List[List[int]] = []
    features: List[np.ndarray] = []

    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode="after")
    def check_len_word_ids(self) -> "TokenizedSet":
        if self.word_ids and len(self.tokens) != len(self.word_ids):
            raise ValueError(
                "The length of `word_ids` and `tokens` must match."
            )
        return self

    @field_validator("word_ids", mode="before")
    @classmethod
    def prepare_word_ids_a(cls, _word_ids):
        new_word_ids = []
        if _word_ids:
            for word_ids in _word_ids:
                if word_ids[0] is None:
                    word_ids[0] = -1
                if word_ids[-1] is None:
                    word_ids[-1] = max(word_ids[:-1]) + 1
                new_word_ids.append(word_ids)
            return new_word_ids
        else:
            return []


class PositionAlignment(BaseModel):
    """
    Alignment between a position in a text,
    and a list of positions in another text.
    """

    position_a: int
    positions_b: List[int]


class TokenAlignment(BaseModel):
    """
    Alignment between a token in a text,
    and a list of tokens in another text.
    """

    token_a: str
    tokens_b: List[str]


class Alignment(BaseModel):
    """
    Alignment of two texts, including the position
    and the tokens of each alignment.
    """

    positions: List[PositionAlignment]
    tokens: List[TokenAlignment]

    def __iter__(self):
        for position_alignment in self.positions:
            yield (
                position_alignment.position_a,
                position_alignment.positions_b,
            )

    def __tokens__(self):
        for token_alignment in self.tokens:
            yield (token_alignment.token_a, token_alignment.tokens_b)

    def merge(self, alignment: "Alignment"):
        self.positions += alignment.positions
        self.tokens += alignment.tokens
