#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <stdbool.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>


/**
 * Fills array with 0 and 1s randomly
*/
void generate_random_ones(uint32_t *, size_t);

/**
 * Fills array half with Gaussian and half with Lorenzian noise
 * Goal is that the output of the masking kernel would be half ones and half zeroes
*/
void generate_noise_array(uint32_t*, size_t);

/**
 * Mysterious function
*/
__device__ float d_M_func(float);

/**
 * Mysterious function
*/
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
__global__ void mask(uint32_t *, uint32_t *, float *, float *, size_t, size_t, size_t, size_t, float, float, float, float*, float*, float*);