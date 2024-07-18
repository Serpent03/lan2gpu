#include "../include/common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* for the sake of simplicity, all ops wil be done in float, and then can be
 * converted at the python end if needed. */

__declspec(dllexport) float **transpose(float **mat, float r, float c) {
  /* call the CUDA entry point from here. */
  float **res = (float **)malloc(
      c * sizeof(float *)); // cudamalloc()  the res. deal with it over there
                            // and then return it back to RAM.
  for (int i = 0; i < c; i++) {
    res[i] = (float *)malloc(r * sizeof(float));
  }
  for (int i = 0; i < c; i++) {
    for (int j = 0; j < r; j++) {
      res[i][j] = mat[j][i];
    }
  }
  return res;
}

__declspec(dllexport) float **matadd(float **mat1, float **mat2, int r, int c) {
  float **res = (float **)malloc(r * sizeof(float *));
  /* this function will basically flatten down the 2D array as a single row
   * matrice */
  float *b1, *b2, *cudaRes;
  b1 = (float *)malloc(r * c * sizeof(float));
  b2 = (float *)malloc(r * c * sizeof(float));
  cudaRes = (float *)malloc(r * c * sizeof(float));

  for (int i = 0; i < r; i++) {
    memcpy(&b1[i * c], mat1[i], c * sizeof(float));
    memcpy(&b2[i * c], mat2[i], c * sizeof(float));
  }

  cuda_mat_add(b1, b2, cudaRes, r, c);
  free(b1);
  free(b2);

  /* after the GPU operation has been done, just rebuild the structure again. */
  for (int i = 0; i < r; i++) {
    res[i] = (float *)malloc(c * sizeof(float));
    memcpy(res[i], &cudaRes[i * c], c * sizeof(float));
  }
  free(cudaRes);
  return res;
}

/* as I can't really assign constant runtime arrays, I will have to deal with
 * malloc() and free(). */
__declspec(dllexport) void free_memory(int **mat, int r) {
  for (int i = 0; i < r; i++) {
    free(mat[i]);
  }
  free(mat);
}

__declspec(dllexport) int prod(int a, int b) { return a * b; }
