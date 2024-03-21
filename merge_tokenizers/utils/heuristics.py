from ..types import Alignment, PositionAlignment, TokenAlignment, TokenizedPair


def align_one_to_one(tokenized_pair: TokenizedPair) -> Alignment:
    """
    Align the tokens of a tokenized pair one to one. This heuristic can only be used
    when the tokenizations match, and can be matched token per token.

    Args:
        tokenized_pair (TokenizedPair): a tokenized pair.

    Returns:
        Alignment: alignment between two tokenizations.
    """
    position_alignment = [
        PositionAlignment(position_a=idx, positions_b=[idx])
        for idx in range(len(tokenized_pair.tokens_a))
    ]
    token_alignment = [
        TokenAlignment(token_a=token, tokens_b=[token])
        for token in tokenized_pair.tokens_a
    ]
    return Alignment(positions=position_alignment, tokens=token_alignment)
