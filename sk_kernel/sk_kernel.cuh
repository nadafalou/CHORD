#include <stdint.h>
#include <stdio.h>
#include <time.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort =
true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
                line);
        if (abort)
            exit(code);
    }
}

__device__ float square(float);
__device__ float cmplx_square(float, float);
__device__ float cmplx_tesseract(float, float);
__global__ void downsample(uint32_t*, float4*, float4*, float4*, float4*, size_t, size_t, size_t, size_t, size_t);
