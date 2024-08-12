#include "../include/common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* for the sake of simplicity, all ops wil be done in float, and then can be
 * converted at the python end if needed. */

void flatten_mat(float **mat, float *out, int r, int c) {
  for (int i = 0; i < r; i++) {
    memcpy(&out[i * c], mat[i], c * sizeof(float));
  }
}

void inflate_mat(float *buf, float **out, int r, int c) {
  for (int i = 0; i < r; i++) {
    out[i] = (float *)malloc(c * sizeof(float));
    memcpy(out[i], &buf[i * c], c * sizeof(float));
  }
}

__declspec(dllexport) float **transpose(float **mat, int r, int c) {
  /* call the CUDA entry point from here. */
  float **res = (float **)malloc(
      c * sizeof(float *)); // cudamalloc()  the res. deal with it over there
                            // and then return it back to RAM.

  float *b1, *cudaRes;
  b1 = (float *)malloc(r * c * sizeof(float));
  cudaRes = (float *)malloc(r * c * sizeof(float));

  flatten_mat(mat, b1, r, c);            // flatten mat[][] into b1[]
  cuda_mat_transpose(b1, cudaRes, r, c); // call CUDA
  free(b1);

  // invert rows and cols, because we've just transposed the array.
  inflate_mat(cudaRes, res, c, r);
  free(cudaRes);

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

  flatten_mat(mat1, b1, r, c);
  flatten_mat(mat2, b2, r, c);
  cuda_mat_add(b1, b2, cudaRes, r, c);
  free(b1);
  free(b2);

  /* after the GPU operation has been done, just rebuild the structure again. */
  inflate_mat(cudaRes, res, r, c);
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
