import ctypes

lib = ctypes.CDLL('./libflush_cache.so')

lib.flush_cache.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
lib.flush_cache.restype = None

def flush_cache_clflush(arr):
    ptr = arr.ctypes.data_as(ctypes.c_void_p)
    size = arr.nbytes
    lib.flush_cache(ptr, size)
