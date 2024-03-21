import numpy as np
from transformers import AutoTokenizer

from merge_tokenizers import (
    DTWAligner,
    FastDTWAligner,
    GreedyAligner,
    PythonDTWAligner,
    TamuheyAligner,
    WordIdsAligner,
)
from merge_tokenizers.types import TokenizedSet

# Load tokenizers.
tokenizer_1 = AutoTokenizer.from_pretrained("roberta-base")
tokenizer_2 = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenizer_3 = AutoTokenizer.from_pretrained("openai-community/gpt2")

# Tokenize a text.
text = "this is how preprocesssing wooorks"
tokenized_1 = tokenizer_1([text])
tokenized_2 = tokenizer_2([text])
tokenized_3 = tokenizer_3([text])

# Get the tokens of a text.
tokens_1 = tokenized_1.tokens()
tokens_2 = tokenized_2.tokens()
tokens_3 = tokenized_3.tokens()

print(tokens_1)
print(tokens_2)
print(tokens_3)

# Get word_ids (only tokenizers that allow it)
# which can be used only in aligners based on word_ids
# like `WordIdsAligner`
word_ids_1 = tokenized_1.word_ids()
word_ids_2 = tokenized_2.word_ids()
word_ids_3 = tokenized_3.word_ids()

# Here we have features associated to each token,
# e.g., token probabilities or hidden states.
dim = 3
features_1 = np.random.rand(len(tokens_1), dim)
features_2 = np.random.rand(len(tokens_2), dim)
features_3 = np.random.rand(len(tokens_3), dim)

# Let's align two tokenized texts with the aligner based on DTW.
aligner = DTWAligner("levenshtein")
alignment = aligner.align(TokenizedSet(tokens=[tokens_1, tokens_2]))[0]
print(list(alignment))
# > [(0, [0]), (1, [1]), (2, [2]), (3, [3]), (4, [4, 5]), (5, [6]), (6, [7]), (7, [8]), (8, [9, 10]), (9, [11])]
print(list(alignment.__tokens__()))
# > [('<s>', ['[CLS]']), ('this', ['this']), ('Ġis', ['is']), ('Ġhow', ['how']), ('Ġpre', ['prep', '##ro']), ('process', ['##ces']), ('sing', ['##ssing']), ('Ġwoo', ['woo']), ('orks', ['##or', '##ks']), ('</s>', ['[SEP]'])]

# We can align also tokenized texts from multiple different tokenizers, using the first tokenizer as reference.
alignments = aligner.align(TokenizedSet(tokens=[tokens_1, tokens_2, tokens_3]))
print([list(alignment) for alignment in alignments])
# > [[(0, [0]), (1, [1]), (2, [2]), (3, [3]), (4, [4, 5]), (5, [6]), (6, [7]), (7, [8]), (8, [9, 10]), (9, [11])], [(0, [0]), (1, [0]), (2, [1]), (3, [2]), (4, [3]), (5, [4]), (6, [5]), (7, [6]), (8, [7]), (9, [7])]]
print([list(alignment.__tokens__()) for alignment in alignments])
# > [[('<s>', ['[CLS]']), ('this', ['this']), ('Ġis', ['is']), ('Ġhow', ['how']), ('Ġpre', ['prep', '##ro']), ('process', ['##ces']), ('sing', ['##ssing']), ('Ġwoo', ['woo']), ('orks', ['##or', '##ks']), ('</s>', ['[SEP]'])], [('<s>', ['this']), ('this', ['this']), ('Ġis', ['Ġis']), ('Ġhow', ['Ġhow']), ('Ġpre', ['Ġpre']), ('process', ['process']), ('sing', ['sing']), ('Ġwoo', ['Ġwoo']), ('orks', ['orks']), ('</s>', ['orks'])]]

# We can merge features associated to each tokenization.
# With `stack` = True we can stack all the features in a numpy array,
# otherwise the result will be a list of features after aligning tokens.
aggregated_features = aligner.aggregate_features(
    TokenizedSet(
        tokens=[tokens_1, tokens_2], features=[features_1, features_2]
    ),
    stack=True,
)
assert (
    aggregated_features.shape[0] == len(tokens_1)
    and aggregated_features.shape[1] == dim * 2
)
print(aggregated_features)

# And the same for multiple tokenized texts.
aggregated_features = aligner.aggregate_features(
    TokenizedSet(
        tokens=[tokens_1, tokens_2, tokens_3],
        features=[features_1, features_2, features_3],
    ),
    stack=True,
)
assert (
    aggregated_features.shape[0] == len(tokens_1)
    and aggregated_features.shape[1] == dim * 3
)

# Let's test all the aligners
dtw_aligner = DTWAligner(distance_name="levenshtein")
dtw_py_aligner = PythonDTWAligner(distance_name="levenshtein")
greedy_aligner = GreedyAligner(distance_name="levenshtein")
fastdtw_aligner = FastDTWAligner(distance_name="euclidean")
tamuhey_aligner = TamuheyAligner()
word_ids_aligner = WordIdsAligner()

aligned_dtw = dtw_aligner.align(TokenizedSet(tokens=[tokens_1, tokens_2]))[0]
aligned_py_dtw = dtw_py_aligner.align(
    TokenizedSet(tokens=[tokens_1, tokens_2])
)[0]
aligned_greedy = greedy_aligner.align(
    TokenizedSet(tokens=[tokens_1, tokens_2])
)[0]
aligned_fastdtw = fastdtw_aligner.align(
    TokenizedSet(tokens=[tokens_1, tokens_2])
)[0]
aligned_tamuhey = tamuhey_aligner.align(
    TokenizedSet(tokens=[tokens_1, tokens_2])
)[0]
aligned_word_ids = word_ids_aligner.align(
    TokenizedSet(tokens=[tokens_1, tokens_2], word_ids=[word_ids_1, word_ids_2])
)[0]

print("C-DTW:", list(aligned_dtw.__tokens__()))
print("PY-DTW:", list(aligned_py_dtw.__tokens__()))
print("Greedy:", list(aligned_greedy.__tokens__()))
print("FastDTW:", list(aligned_fastdtw.__tokens__()))
print("Tamuhey:", list(aligned_tamuhey.__tokens__()))
print("WordIds:", list(aligned_word_ids.__tokens__()))
