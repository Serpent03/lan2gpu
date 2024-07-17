import ctypes

# Load the shared library
computation = ctypes.CDLL('./build/comp.dll', winmode = 0)

# Define argument and return types
computation.transpose.argtypes = (ctypes.POINTER(
    ctypes.POINTER(ctypes.c_int)), ctypes.c_int32, ctypes.c_int32)
computation.transpose.restype = ctypes.POINTER(ctypes.POINTER(ctypes.c_int))
computation.free_memory_mat.argtypes = (
    ctypes.POINTER(ctypes.POINTER(ctypes.c_int)), ctypes.c_int)


def list_to_ctypes_2d(arr):
    r, c = len(arr), len(arr[0])
    # creating a cytpes object is the most pain of them all.
    arr_type = ctypes.POINTER(ctypes.c_int) * r
    c_arr = arr_type()
    for i in range(r):
        row = (ctypes.c_int * c)(*arr[i])
        c_arr[i] = ctypes.cast(row, ctypes.POINTER(ctypes.c_int))
    return c_arr, r, c


def mat_transpose(arr):
    c_arr, r, c = list_to_ctypes_2d(arr)
    result = computation.transpose(c_arr, r, c)
    transposed_arr = []
    for i in range(c):
        transposed_arr.append([result[i][j] for j in range(r)])
    computation.free_memory_mat(result, r)
    return transposed_arr

def prod(a, b):
    computation.prod.argtypes = (ctypes.c_int, ctypes.c_int)
    computation.prod.restype = (ctypes.c_int)
    return computation.prod(a, b)


mat = [[1, 1, 1], [0, 0, 0]]
# result = mat_transpose(mat)
# for i in mat:
#     print(i)
# for i in result:
#     print(i)
print(prod(51, 2))
