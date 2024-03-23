// https://www.digitalocean.com/community/tutorials/calling-c-functions-from-python
// cc -fPIC -shared -o merge_tokenizers/aligners/dtw_c/dtw.so merge_tokenizers/aligners/dtw_c/dtw.c

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>

typedef struct {
    int first;
    int second;
} Tuple;

typedef struct {
    Tuple* alignment;
    int n_elements;
} AlignmentResult;

AlignmentResult dtw_alignment(int len_a, int len_b, int* distances, int radius) {
    // Compute distance matrix
    int** matrix = (int**)malloc((len_a + 1) * sizeof(int*));
    for (int i = 0; i <= len_a; i++) {
        matrix[i] = (int*)malloc((len_b + 1) * sizeof(int));
    }

    for (int i = 0; i <= len_a; i++) {
        for (int j = 0; j <= len_b; j++) {
            matrix[i][j] = INT_MAX;
        }
    }
    matrix[0][0] = 0;
    int dist = 0;
    for (int i = 1; i < len_a; i++) {
        for (int j = 1; j < len_b; j++) {
            dist = distances[i * len_b + j];
            if (radius > 0) {
                if (abs(i-j) <= radius){
                    matrix[i][j] = fminf(matrix[i - 1][j], fminf(matrix[i - 1][j - 1], matrix[i][j - 1])) + dist;
                }
            }
            else {
                matrix[i][j] = fminf(matrix[i - 1][j], fminf(matrix[i - 1][j - 1], matrix[i][j - 1])) + dist;
            } 
        }
    }
    // Recover pointers
    int i = len_a, j = len_b;
    Tuple* alignment = (Tuple*)malloc((len_a + len_b) * sizeof(Tuple));
    int index = 0;
    float min_ = 0;
    while (i > 0 && j > 0) {
        min_ = fminf(matrix[i - 1][j], fminf(matrix[i][j - 1], matrix[i - 1][j - 1]));
        if (min_ == matrix[i - 1][j]) {
            alignment[index].first = i - 1;
            alignment[index].second = j;
            i--;
        }
        else if (min_ == matrix[i][j - 1]) {
            alignment[index].first = i;
            alignment[index].second = j - 1;
            j--;
        } else {
            alignment[index].first = i - 1;
            alignment[index].second = j - 1;
            i--;
            j--;
        }
        index++;
    }
    AlignmentResult result;
    result.alignment = alignment;
    result.n_elements = index;
    return result;
}

void free_alignment_result(AlignmentResult result) {
    free(result.alignment);
}