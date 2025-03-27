// flush_cache.c
#include <emmintrin.h> // _mm_clflush
#include <stdint.h>

void flush_cache(void* addr, size_t size) {
    for (size_t i = 0; i < size; i += 64) { 
        _mm_clflush((char*)addr + i);
    }
    _mm_mfence(); 
}
