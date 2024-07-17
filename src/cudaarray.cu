#include "../include/common.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void _cuda_mat_add(int **mat, int r, int c) {
  /* now do some stuff here */
}

extern "C" {
void cuda_entry(void *buffer1, void *buffer2, size_t x1, size_t y1, size_t x2,
                size_t y2, enum TYPE type) {
  printf("Hello from the GPU! a: %d, b: %d\n", (int)x1, (int)y1);
  switch (type) {
  case INT:
    printf("INT detected!\n");
    break;
  default:
    printf("reached default\n");
    break;
  }
  _cuda_mat_add<<<1, 1>>>(NULL, x1, y1);
}
}