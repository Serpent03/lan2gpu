#include "common.h"
#include <stdio.h>
#include <stdlib.h>

extern void cuda_entry(void *b1, void *b2, size_t x1, size_t y1, size_t x2,
                       size_t y2, enum TYPE type);

__declspec(dllexport) int **transpose(int **mat, int r, int c) {
  int **res = (int **)malloc(
      c * sizeof(int *)); // cudamalloc()  the res. deal with it over there and
                          // then return it back to RAM.
  for (int i = 0; i < c; i++) {
    res[i] = (int *)malloc(r * sizeof(int));
  }
  for (int i = 0; i < c; i++) {
    for (int j = 0; j < r; j++) {
      res[i][j] = mat[j][i];
    }
  }
  return res;
}

__declspec(dllexport) void free_memory_mat(int **mat, int r) {
  for (int i = 0; i < r; i++) {
    free(mat[i]);
  }
  free(mat);
}

__declspec(dllexport) int prod(int a, int b) {
  cuda_entry(NULL, NULL, a, b, 0, 0, INT);
  return a * b;
}
