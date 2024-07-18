#include "../include/common.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

__global__ void _cuda_mat_add(float *mat1, float *mat2, float *out,
                              size_t size) {
  int i = blockDim.x * blockIdx.x +
          threadIdx.x; /* big things come in small measures */
  if (i < size) {
    out[i] = mat1[i] + mat2[i];
  }
}

extern "C" {
void cuda_mat_add(float *mat1, float *mat2, float *res, int r, int c) {
  /* d_a and d_b are both the input buffers. d_c is generated in the GPU memory
  and copied over to res, which is then returned for further processing. */
  float *d_a, *d_b, *d_out;
  cudaMalloc((void **)&d_a, r * c * sizeof(float));
  cudaMalloc((void **)&d_b, r * c * sizeof(float));
  cudaMalloc((void **)&d_out, r * c * sizeof(float));

  cudaMemcpy(d_a, mat1, r * c * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, mat2, r * c * sizeof(float), cudaMemcpyHostToDevice);

  _cuda_mat_add<<<1, r * c>>>(d_a, d_b, d_out, r * c);

  cudaMemcpy(res, d_out, r * c * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_out);
}
}