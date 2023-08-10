#include <stdint.h>
#include <stdio.h>
#include <time.h>

__device__ float square(float);
__device__ float cmplx_square(float, float);
__device__ float cmplx_tesseract(float, float);
__global__ void downsample(uint32_t*, float4*, float4*, float4*, float4*, size_t, size_t, size_t, size_t, size_t);
