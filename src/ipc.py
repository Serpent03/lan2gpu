import ctypes
import traceback
from typing import Union


class ipc():
    def __init__(self, dllpath: str) -> None:
        self.lib = ctypes.CDLL(dllpath)

    def list_to_ctypes_2d(self, arr) -> Union[list[list[float]], int, int]:
        # creating a cytpes object is the most pain of them all...
        r, c = len(arr), len(arr[0])
        arr_type = ctypes.POINTER(ctypes.c_float) * r
        c_arr = arr_type()
        for i in range(r):
            row = (ctypes.c_float * c)(*arr[i])
            c_arr[i] = ctypes.cast(row, ctypes.POINTER(ctypes.c_float))
        return c_arr, r, c

    def transpose(self, buf1: list[list[float]]) -> list[list[float]]:
        self.lib.transpose.argtypes = (ctypes.POINTER(
            ctypes.POINTER(ctypes.c_float)), ctypes.c_int, ctypes.c_int)
        self.lib.transpose.restype = (
            ctypes.POINTER(ctypes.POINTER(ctypes.c_float)))
        buf1, r, c = self.list_to_ctypes_2d(buf1)
        res = self.lib.transpose(buf1, r, c)
        final_array = []
        for i in range(c):
            # since the result is transposed, we iterated on c and r, instead of r and c.
            final_array.append([res[i][j] for j in range(r)])
        self.__free(res, r)
        return final_array

    def mat_add(self, buf1: list[list[float]], buf2: list[list[float]]) -> list[list[float]]:
        # dimensions of both buf1 and buf2 **MUST** be the same for matrix addition.
        try:
            assert len(buf1) == len(buf2) and len(buf1[0]) == len(
                buf2[0]), "Buffers have unequal dimensions"
        except Exception as e:
            return f"RuntimeException: {e}"
        self.lib.matadd.argtypes = (ctypes.POINTER(ctypes.POINTER(ctypes.c_float)), ctypes.POINTER(
            ctypes.POINTER(ctypes.c_float)), ctypes.c_int, ctypes.c_int)
        self.lib.matadd.restype = (
            ctypes.POINTER(ctypes.POINTER(ctypes.c_float)))
        buf1, r, c = self.list_to_ctypes_2d(buf1)
        # since the dimensions are the same, we don't need r2 and c2
        buf2, _, _ = self.list_to_ctypes_2d(buf2)
        res = self.lib.matadd(buf1, buf2, r, c)
        final_array = []
        for i in range(r):
            final_array.append([res[i][j] for j in range(c)])
        self.__free(res, r)
        return final_array

    def mat_mult(self, buf1: list[list[int]], buf2: list[list[int]]) -> list[list[int]]:
        # buf1, buf2, w1, h1, w2, h2
        self.lib.matmult.argtypes = (ctypes.POINTER(ctypes.POINTER(ctypes.c_float)), ctypes.POINTER(
            ctypes.POINTER(ctypes.c_float)), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int)
        # result, w, h
        self.lib.matmult.restype = (
            ctypes.POINTER(ctypes.POINTER(ctypes.c_float)), )

    def __free(self, buf: any, rows: int) -> None:
        self.lib.free_memory.argtypes = (ctypes.POINTER(
            ctypes.POINTER(ctypes.c_float)), ctypes.c_int)
        self.lib.free_memory.restype = None
        self.lib.free_memory(buf, rows)
