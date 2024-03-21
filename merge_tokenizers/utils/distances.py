from functools import lru_cache
from typing import Callable, List, Union

import numpy as np
import ukkonen
from Levenshtein import distance as levenshtein
from scipy.spatial.distance import cosine, euclidean


@lru_cache(maxsize=None)
def levenshtein_distance(text_a: str, text_b: str) -> int:
    """
    Computes levenshtein distance using the implementation of
    https://github.com/rapidfuzz/Levenshtein and caches results.

    Args:
        text_a (str): text to be compared.
        text_b (str): another text to be compared.

    Returns:
        int: levenshtein distance of both texts.
    """
    return levenshtein(text_a, text_b)


@lru_cache(maxsize=None)
def ukkonen_distance(text_a: str, text_b: str, k: int = 5) -> int:
    """
    Computes ukkonen distance using the implementation of
    https://github.com/asottile/ukkonen and caches results.

    Args:
        text_a (str): text to be compared.
        text_b (str): another text to be compared.

    Returns:
        int: levenshtein distance of both texts.
    """
    return ukkonen.distance(text_a, text_b, k)


@lru_cache(maxsize=None)
def intersection_distance(text_a: str, text_b: str) -> int:
    max_len = max(len(text_a), len(text_b))
    return max_len - len(set(text_a).intersection(set(text_b)))


@lru_cache(maxsize=None)
def cosine_distance(repr_text_a: np.ndarray, repr_text_b: np.ndarray) -> float:
    return cosine(repr_text_a, repr_text_b)


@lru_cache(maxsize=None)
def euclidean_distance(
    repr_text_a: np.ndarray, repr_text_b: np.ndarray
) -> float:
    return euclidean(repr_text_a, repr_text_b)


def precompute_distances(
    texts_a: Union[List[str], np.ndarray],
    texts_b: Union[List[str], np.ndarray],
    distance_fn: Callable,
) -> List[Union[int, float]]:
    """
    Precomputes a distance matrix between the texts in `texts_a` and
    the texts in `texts_b` using the `distance_fn` function.

    Args:
        texts_a (List[str]): list of texts.
        texts_b (List[str]): another list of texts.
        distance_fn (Callable): a distance function.

    Returns:
        List[int]: list of distances between the texts in
                   `texts_a` and the texts in `texts_b`.
    """
    distances = []
    for text_a in texts_a:
        for text_b in texts_b:
            distances.append(distance_fn(text_a, text_b))
    return distances


def get_distance_fn(name: str) -> Callable:
    """
    Returns a distance function from his name.

    Args:
        name (str): name of the function in this module.

    Returns:
        Callable: a function.
    """
    distance_fns = {
        "levenshtein": levenshtein_distance,
        "ukkonen": ukkonen_distance,
        "intersection": intersection_distance,
        "cosine": cosine_distance,
        "euclidean": euclidean_distance,
    }
    return distance_fns.get(name, levenshtein_distance)  # type: ignore
