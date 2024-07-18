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

  /* threadCount limit of ~1024. Can't launch with just that - we can use the 2^31 - 1 blocks instead... :) 
  we first take the total size {r * c}, and because dividing it by 256 might not cover all the cases, add one.
  eg: row, col = 100, 100 => 10000 separate elements
  10000 / 256(threads per block) => ~39
  39 * 256 = 9984 different threads -- which does **not** cover all the separate elements.

  So: 
  => 10000 / 256 + 255 / 256
  => ~(39 + 0.9)
  => ~(40.8)
  => ~40. 40 * 256 = 10240, which is enough to cover all elements. */
  uint16 threadsPerBlock = 256;
  int32 blockCount = (r * c + threadsPerBlock - 1) / threadsPerBlock;
  _cuda_mat_add<<<blockCount, threadsPerBlock>>>(d_a, d_b, d_out, r * c);

  cudaMemcpy(res, d_out, r * c * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_out);
}
}