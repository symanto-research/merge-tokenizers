// https://www.digitalocean.com/community/tutorials/calling-c-functions-from-python
// cc -fPIC -shared -o merge_tokenizers/aligners/greedy_coverage_c/greedy_coverage.so merge_tokenizers/aligners/greedy_coverage_c/greedy_coverage.c

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int start;
    int end;
} Tuple;

typedef struct {
    Tuple* alignment;
    int n_elements;
} AlignmentResult;

Tuple* get_spans(char** tokens, char* text, int tokens_count) {
    Tuple* spans = (Tuple*)malloc(tokens_count * sizeof(Tuple));
    int j = 0;
    for (int k = 0; k < tokens_count; k++) {
        char* token = tokens[k];
        int i = 0;
        int start_pos = -1;
        int end_pos = -1;
        int matches = 0;
        while (token[i] != '\0') {
            if (token[i] == text[j + matches]) {
                if (start_pos == -1) {
                    start_pos = j + matches;
                }
                end_pos = j + matches;
                matches++;
            }
            i++;
        }
        if (end_pos != -1) {
            j = end_pos + 1;
        }
        if (start_pos != -1 && end_pos != -1) {
            spans[k].start = start_pos;
            spans[k].end = end_pos;
            if (j >= strlen(text)) {
                break;
            }
        } else {
            spans[k].start = -1;
            spans[k].end = -1;
        }
    }
    return spans;
}

AlignmentResult merge_spans(Tuple* spans_a, Tuple* spans_b, int spans_a_count, int spans_b_count) {
    Tuple* alignments = (Tuple*)malloc((spans_a_count + spans_b_count) * sizeof(Tuple));
    int i = 0, j = 0, alignment_index = 0;
    while (i < spans_a_count && j < spans_b_count) {
        int a_start = spans_a[i].start;
        int a_end = spans_a[i].end;
        int b_start = spans_b[j].start;
        int b_end = spans_b[j].end;
        alignments[alignment_index].start = i;
        alignments[alignment_index].end = j;
        alignment_index++;

        if (a_start == b_start && a_end == b_end) {
            i++;
            j++;
        } else if (a_end == b_end) {
            i++;
            j++;
        } else if (a_end <= b_end) {
            i++;
        } else {
            j++;
        }
    }

    while (i < spans_a_count) {
        alignments[alignment_index].start = i;
        alignments[alignment_index].end = j - 1;
        alignment_index++;
        i++;
    }

    while (j < spans_b_count) {
        alignments[alignment_index].start = i - 1;
        alignments[alignment_index].end = j;
        alignment_index++;
        j++;
    }

    AlignmentResult result;
    result.alignment = alignments;
    result.n_elements = alignment_index;
    return result;
}

void free_spans(Tuple* tuple_list) {
    free(tuple_list);
}

void free_alignment(AlignmentResult result) {
    free(result.alignment);
}