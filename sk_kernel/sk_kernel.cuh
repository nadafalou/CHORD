#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <stdbool.h>
#include <stdlib.h>
#include <cuda_runtime.h>

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
 * @param stream CUDA stream (if not specified, then default stream will be used)
*/

extern void launch_downsample_kernel(
    const uint32_t *E,   // shape (T, F, D/2)
    float4 *S1,          // shape (T/N, F, D/2)
    float4 *S2,          // shape (T/N, F, D/2)
    float4 *S1_p,        // shape (T/N_p, F, D/2)
    float4 *S2_p,        // shape (T/N_p, F, D/2)
    size_t N,            // first-stage downsampling factor (value TBD, maybe 256 in CHORD)
    size_t N_p,          // second-stage downsampling factor (value TBD, maybe 32*256 in CHORD)
    size_t D,            // number of dishes (64 in CHORD pathfinder, 512 in full CHORD)
    size_t T,            // number of time samples
    size_t F,            // number of frequency channels
    cudaStream_t stream = nullptr   // if not specified, then default cuda stream will be used
);


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
 * Fills matrix with random values
*/
void generate_random(uint32_t *, size_t);

/**
 * Returns square of int
*/
__device__ __forceinline__ int square(int);

/**
 * Returns the square of a complex number
 * 
 * Given a complex numebr <real + i * imaginary>, function returns
 * <real^2 + imaginary^2>
*/
__device__ __forceinline__ int cmplx_square(int, int);

/**
 * Returns the tesseract (4th power) of a complex number
 * 
 * Given a complex number <real + i * imaginary>, function returns
 * <(real^2 + imaginary^2)^2>
*/
__device__ __forceinline__ int cmplx_tesseract(int, int);
