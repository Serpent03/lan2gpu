#pragma once

typedef unsigned char uint8;
typedef char int8;

typedef unsigned short int uint16;
typedef short int int16;

typedef unsigned int uint32;
typedef int int32;

typedef long unsigned int uint64;
typedef long int int64;

enum TYPE { CHAR, UCHAR, USHORTINT, SHORTINT, INT, UINT, FLOAT, DOUBLE };

/**
 * Based on the enum TYPE we can gather what the element size of buffer1 and
 * buffer2 would be.
 * @param buffer1 One of the buffers to pass
 * @param buffer2 One of the buffers to pass
 * @param n1 The length of buffer1
 * @param n2 The length of buffer2
 * @param type The type of element each buffer has
 */
// extern void cuda_entry(void *buffer1, void *buffer2, size_t n1, size_t n2,
// enum TYPE type);
