from ..types import Alignment, PositionAlignment, TokenAlignment, TokenizedPair
from .base import Aligner


class WordIdsAligner(Aligner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kwargs = kwargs

    def _align_pair(
        self,
        tokenized_pair: TokenizedPair,
    ) -> Alignment:
        """
        Aligns the tokens from two different tokenizers, using
        the `word_ids` provided by HuggingFace tokenizers.

        Tries to match all the tokens of the same word from both
        sequences in order, using a modified version of the posting
        list intersection algorithm:
        https://nlp.stanford.edu/IR-book/html/htmledition/processing-boolean-queries-1.html
        """
        # Compute alignments based on word_ids
        alignments = []
        i, j = 0, 0
        while i < len(tokenized_pair.preprocessed_tokens_a) and j < len(
            tokenized_pair.preprocessed_tokens_b
        ):
            word_id_a = tokenized_pair.word_ids_a[i]
            word_id_b = tokenized_pair.word_ids_b[j]
            if word_id_a == word_id_b:
                alignments.append((i, [j]))
                i += 1
                j += 1
            else:
                if word_id_a < word_id_b:
                    alignments.append((i, [j - 1]))
                    i += 1
                else:
                    alignments[-1][1].append(j)
                    j += 1

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
