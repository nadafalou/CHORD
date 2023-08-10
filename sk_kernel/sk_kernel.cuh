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

/**
 * Returns square of float
*/
__device__ float square(float);

/**
 * Returns the square of a complex number
 * 
 * Given a complex numebr <real + i * imaginary>, function returns
 * <real^2 + imaginary^2>
*/
__device__ float cmplx_square(float, float);

/**
 * Returns the tesseract (4th power) of a complex number
 * 
 * Given a complex number <real + i * imaginary>, function returns
 * <(real^2 + imaginary^2)^2>
*/
__device__ float cmplx_tesseract(float, float);

/**
 * Downsamples the "raw" electric field array <E> and writes output to arrays
 * <S1>, <S2>, <S1_p> and <S2_p>
 * 
 * Refer to chord_rfi.pdf for details on downsampling
 * 
 * @param E Input electric field array
 * @param S1 Output squared array downsampled by a factor of N
 * @param S2 Output of tesseracted array downsampled by a factor of N
 * @param S1_p Output squared array downsampled by a factor of N_p
 * @param S2_p Output of tesseracted array downsampled by a factor of N_p
 * @param N Downsampling factor
 * @param N_p Downsampling factor, multiple of N
 * @param D Number of dishes
 * @param T Number of time samples
 * @param F Number of frequency channels
*/
__global__ void downsample(uint32_t*, float4*, float4*, float4*, float4*, size_t, size_t, size_t, size_t, size_t);
