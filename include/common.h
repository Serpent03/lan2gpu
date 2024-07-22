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
extern "C" void cuda_mat_add(float *mat1, float *mat2, float *res, int r, int c);
extern "C" void cuda_mat_transpose(float *mat, float *res, int r, int c);
#else
/**
 * Call the CUDA wrapper for adding matrices.
 * @param mat1 One of the buffers to pass
 * @param mat2 One of the buffers to pass
 * @param res The resulting buffer
 * @param r Number of rows
 * @param c Number of columns
 */
extern void cuda_mat_add(float *mat1, float *mat2, float *res, int r, int c);
/**
 * Call the CUDA wrapper for transposing matrices.
 * @param mat One of the buffers to pass
 * @param res The resulting buffer
 * @param r Number of rows
 * @param c Number of columns
 */
extern void cuda_mat_transpose(float *mat, float *res, int r, int c);
#endif

/* since C and C++(CUDA C++) import headers in a different fashion. */
