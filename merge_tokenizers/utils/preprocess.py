import unicodedata
from typing import Callable, List


def normalize_unicode(text: str, normalization: str = "NFKD") -> str:
    """
    Normalizes unicode with NFKD as default to obtain
    canonical forms without compatibility information.

    Args:
        text (str): a text
        normalization (str): unicode normalization

    Returns:
        str: unicode-normalized text
    """
    return unicodedata.normalize(normalization, text)


def lowercase(text: str) -> str:
    """
    Wrapper to do lowercase.

    Args:
        text (str): a text

    Returns:
        str: a lowercased text
    """
    return text.lower()


def preprocess_tokens(tokens: List[str]) -> List[str]:
    """
    Function to preprocess tokens.

    Args:
        tokens (List[str]): list of tokens

    Returns:
        List[str]: preprocessed tokens
    """
    pipeline: List[Callable] = [normalize_unicode, lowercase]
    preprocessed = []
    for token in tokens:
        for fn in pipeline:
            token = fn(token)
        preprocessed.append(token)
    return preprocessed
