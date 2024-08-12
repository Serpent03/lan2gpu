#pragma once

typedef unsigned char uint8;
typedef char int8;

typedef unsigned short int uint16;
typedef short int int16;

typedef unsigned int uint32;
typedef int int32;

typedef long unsigned int uint64;
typedef long int int64;

#ifdef __cplusplus
extern "C" void cuda_mat_add(float *mat1, float *mat2, float *res, int r,
                             int c);
extern "C" void cuda_mat_transpose(float *mat, float *res, int r, int c);
#else
/* Call CUDA to add two matrices MAT1 and MAT2, with rows R and columns C, and
 * return the result in RES. */
extern void cuda_mat_add(float *mat1, float *mat2, float *res, int r, int c);
/* Call CUDA to transpose a matrice MAT, with rows R and columns C, and return
 * the result in RES. */
extern void cuda_mat_transpose(float *mat, float *res, int r, int c);
#endif

/* since C and C++(CUDA C++) import headers in a different fashion. */
