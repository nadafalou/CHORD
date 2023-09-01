#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <stdbool.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>


/**
 * Fills matrix with 0 and 1s randomly
*/
void generate_random_ones(bool *, size_t);

/**
 * Fills matrix with random floats
*/
void generate_random_float4(float4 *, size_t);

__device__ float d_M_func(float);

__device__ float d_V_func(float);

/**
 * Mask 
 * 
 * Refer to chord_rfi.pdf for details on downsampling
 * 
 * @param R Output masking array
 * @param W Input bad feed mask
 * @param S1 Output squared array downsampled by a factor of N
 * @param S2 Output of tesseracted array downsampled by a factor of N
 * @param N Downsampling factor
 * @param D Number of dishes
 * @param T_bar Number of course time samples
 * @param F Number of frequency channels
 * @param mu_min Minimum for mu to be considered good
 * @param N_good_min Minimum number of good feeds to continue
*/
__global__ void mask(bool *, bool *, float4 *, float4 *, size_t, size_t, size_t, size_t, float, float);