#include "sk_kernel.cuh"
#include <stdexcept>  // std::runtime_error

void generate_random(uint32_t *arr, size_t size) {
    srand(time(NULL)); 
    for (int i = 0; i < size; i++) {
        uint32_t e = rand() << 28;
        e = e ^ ((rand() & 0xf) << 24);
        e = e ^ ((rand() & 0xf) << 20);
        e = e ^ ((rand() & 0xf) << 16);
        e = e ^ ((rand() & 0xf) << 12);
        e = e ^ ((rand() & 0xf) << 8);
        e = e ^ ((rand() & 0xf) << 4);
        e = e ^ (rand() & 0xf);

        arr[i] = (uint32_t) e;
    }
}

__device__ __forceinline__ int square(int num) {
    return num * num;
}

__device__ __forceinline__ int cmplx_square(int real, int imaginary) {
    return square(real) + square(imaginary);
}

__device__ __forceinline__ int cmplx_tesseract(int real, int imaginary) { 
    return square(cmplx_square(real, imaginary));
}

__device__ void s_update(float &s1, float &s2, float &s1_p, float &s2_p, float re, float im) {
    float sq = cmplx_square(re, im);
    s1 += sq;
    s1_p += sq;
    s2 += sq*sq;
    s2_p += sq*sq;
}

__device__ void store_float4(float4 *p, float x, float y, float z, float w) {
    float4 tmp;
    tmp.x = x;
    tmp.y = y;
    tmp.z = z;
    tmp.w = w;
    *p = tmp;
}

__global__ void __launch_bounds__(128, 4) downsample(const uint32_t *E, float4 *S1, float4 *S2, float4 *S1_p, float4 *S2_p, size_t N, size_t N_p, size_t D, size_t T, size_t F) {
    // if each thread took one time sample, not N':
    int num_threads;
    D == 64 ? num_threads = 32 : num_threads = 32 * 4;

    int s1_0, s1_1, s1_2, s1_3, s2_0, s2_1, s2_2, s2_3;
    s1_0 = s1_1 = s1_2 = s1_3 = s2_0 = s2_1 = s2_2 = s2_3 = 0;
    int s1_p_0, s1_p_1, s1_p_2, s1_p_3, s2_p_0, s2_p_1, s2_p_2, s2_p_3;
    s1_p_0 = s1_p_1 = s1_p_2 = s1_p_3 = s2_p_0 = s2_p_1 = s2_p_2 = s2_p_3 = 0;
    int e0_re, e0_im, e1_re, e1_im, e2_re, e2_im, e3_re, e3_im;
    e0_re = e0_im = e1_re = e1_im = e2_re = e2_im = e3_re = e3_im = 0;

    int n_p;
    int n = 0;

    // optimisation did not make a difference 
    // E += F * D / 2 * N_p * blockIdx.z + D / 2 * blockIdx.x + 32 * blockIdx.y + threadIdx.x;

    size_t S_index = (F * (D / 2) * (N_p / N) * blockIdx.z) + ((D / 2) * blockIdx.x) + (num_threads * blockIdx.y) + threadIdx.x;
    S1 += S_index;
    S2 += S_index;

    // Sum up data of N' time samples
    for (n_p = 0; n_p < N_p; n_p++) {
        // Get 4 feeds (packed into 1 uint32)
        uint32_t e = E[F * D / 2 * N_p * blockIdx.z + F * D / 2 * n_p + D / 2 * blockIdx.x + num_threads * blockIdx.y + threadIdx.x];
        
        // uint32_t e = E[F * D / 2 * n_p];

        // Unpack uint32 into 4 complex numbers (each with real and imaginary components)
        e0_re = int(e & 0xf);
        e0_im = int((e >> 4) & 0xf);
        e1_re = int((e >> 8) & 0xf);
        e1_im = int((e >> 12) & 0xf);
        e2_re = int((e >> 16) & 0xf);
        e2_im = int((e >> 20) & 0xf);
        e3_re = int((e >> 24) & 0xf);
        e3_im = int((e >> 28) & 0xf);

        // Square/tesseract and sum
        s1_0 += cmplx_square(e0_re, e0_im);
        s1_1 += cmplx_square(e1_re, e1_im);
        s1_2 += cmplx_square(e2_re, e2_im);
        s1_3 += cmplx_square(e3_re, e3_im);

        s2_0 += cmplx_tesseract(e0_re, e0_im);
        s2_1 += cmplx_tesseract(e1_re, e1_im);
        s2_2 += cmplx_tesseract(e2_re, e2_im);
        s2_3 += cmplx_tesseract(e3_re, e3_im);

        s1_p_0 += cmplx_square(e0_re, e0_im);
        s1_p_1 += cmplx_square(e1_re, e1_im);
        s1_p_2 += cmplx_square(e2_re, e2_im);
        s1_p_3 += cmplx_square(e3_re, e3_im);

        s2_p_0 += cmplx_tesseract(e0_re, e0_im);
        s2_p_1 += cmplx_tesseract(e1_re, e1_im);
        s2_p_2 += cmplx_tesseract(e2_re, e2_im);
        s2_p_3 += cmplx_tesseract(e3_re, e3_im);

        // this optimisation did not make a difference
        // s_update(s1_0, s2_0, s1_p_0, s2_p_0, e0_re, e0_im);
        // s_update(s1_1, s2_1, s1_p_1, s2_p_1, e1_re, e1_im);
        // s_update(s1_2, s2_2, s1_p_2, s2_p_2, e2_re, e2_im);
        // s_update(s1_3, s2_3, s1_p_3, s2_p_3, e3_re, e3_im);

        // if n is a factor of N (the smaller downsampling factor), write out 
        // to S1 and S2, and reset s1, s2 back to 0
        n++;
        if (n == N) {
            // size_t write_index = (F * (D / 2) * (N_p / N) * blockIdx.z) + (F * (D / 2) * ((n - 1) / N)) + ((D / 2) * blockIdx.x) + (32 * blockIdx.y) + threadIdx.x;

            S1[0].x = s1_0;
            S1[0].y = s1_1;
            S1[0].z = s1_2;
            S1[0].w = s1_3;

            S2[0].x = s2_0;
            S2[0].y = s2_1;
            S2[0].z = s2_2;
            S2[0].w = s2_3;

            // this optimisation did not make a difference
            // store_float4(S1, s1_0, s1_1, s1_2, s1_3);
            // store_float4(S2, s2_0, s2_1, s2_2, s2_3);

            s1_0 = s1_1 = s1_2 = s1_3 = 0;
            s2_0 = s2_1 = s2_2 = s2_3 = 0;

            S1 += F * (D/2);
            S2 += F * (D/2);
            n = 0;
        }
    }

    // write out to S1' and S2'
    // optimisation did not make a difference
    size_t write_index = F * D / 2 * blockIdx.z + D / 2 * blockIdx.x + num_threads * blockIdx.y + threadIdx.x;
    S1_p[write_index].x = s1_p_0;
    S1_p[write_index].y = s1_p_1;
    S1_p[write_index].z = s1_p_2;
    S1_p[write_index].w = s1_p_3;

    S2_p[write_index].x = s2_p_0;
    S2_p[write_index].y = s2_p_1;
    S2_p[write_index].z = s2_p_2;
    S2_p[write_index].w = s2_p_3;
}



void launch_downsample_kernel(const uint32_t *E, float4 *S1, float4 *S2, float4 *S1_p, float4 *S2_p, size_t N, size_t N_p, size_t D, size_t T, size_t F, cudaStream_t stream)
{
    // Define some reasonable ranges for integer-valued arguments.
    if ((F <= 0) || (F > 10000))
	throw std::runtime_error("launch_downsample_kernel(): invalid value of F");
    if ((D <= 0) || (D > 10000))
	throw std::runtime_error("launch_downsample_kernel(): invalid value of D");
    if ((T <= 0) || (T > 10*1000*1000))
	throw std::runtime_error("launch_downsample_kernel(): invalid value of T");
    if ((N <= 0) || (N > 10*1000*1000))
	throw std::runtime_error("launch_downsample_kernel(): invalid value of N");
    if ((N_p <= 0) || (N_p > 10*1000*1000))
	throw std::runtime_error("launch_downsample_kernel(): invalid value of N_p");

    // Check for null pointers. (Unfortunately this doesn't catch the common error of passing a host pointer
    // instead of a device pointer, but I don't think there's a way to check for that!)

    if (!E || !S1 || !S2 || !S1_p || !S2_p)
	throw std::runtime_error("launch_downsample_kernel(): null array pointer was specified");

    // Some nontrivial parameter checks.
    if (N_p % N)
	throw std::runtime_error("launch_downsample_kernel(): N_p must be a multiple of N");
    if (T % N_p)
	throw std::runtime_error("launch_downsample_kernel(): T must be a multiple of N_p");
    if ((D != 64) && (D % 256))
	throw std::runtime_error("launch_downsample_kernel(): invalid value of D (kernel currently supports 64 or multiple of 256)");

    // This restriction arises because we set blocks.z = (T/N_p), and 65535 is the max value allowed by cuda.
    // (Reference: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications)
    if (T > 65535*N_p)
	throw std::runtime_error("launch_downsample_kernel(): (T/N_p) cannot be larger than 65535");
	 
    // Each warp processes 128 feeds (i.e. dish-polarization pairs).
    // If D==64 (i.e. 128 feeds), then we use one warp per threadblock.
    // If D is a multiple of 256, then we use four warps per threadblock (and set blocks.y = (D/256)).
    int threads = (D==64) ? 32 : 128;
    
    dim3 blocks;
    blocks.x = F;
    blocks.y = (D==64) ? 1 : (D/256);
    blocks.z = T/N_p;

    downsample <<< blocks, threads, 0, stream >>> (E, S1, S2, S1_p, S2_p, N, N_p, D, T, F);
    cudaError_t err = cudaPeekAtLastError();
    
    if (err != cudaSuccess)
	throw std::runtime_error("launch_downsample_kernel(): kernel launch failed!");
}
