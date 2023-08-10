#include "sk_kernel.cuh"

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

__device__ float square(float num) {
    return num * num;
}

__device__ float cmplx_square(float real, float imaginary) {
    return square(real) + square(imaginary);
}

__device__ float cmplx_tesseract(float real, float imaginary) { 
    return square(cmplx_square(real, imaginary));
}

__global__ void downsample(uint32_t *E, float4 *S1, float4 *S2, float4 *S1_p, float4 *S2_p, size_t N, size_t N_p, size_t D, size_t T, size_t F) {
    // if each thread took one time sample, not N':

    float s1_0, s1_1, s1_2, s1_3, s2_0, s2_1, s2_2, s2_3;
    s1_0 = s1_1 = s1_2 = s1_3 = s2_0 = s2_1 = s2_2 = s2_3 = 0;
    float s1_p_0, s1_p_1, s1_p_2, s1_p_3, s2_p_0, s2_p_1, s2_p_2, s2_p_3;
    s1_p_0 = s1_p_1 = s1_p_2 = s1_p_3 = s2_p_0 = s2_p_1 = s2_p_2 = s2_p_3 = 0;
    float e0_re, e0_im, e1_re, e1_im, e2_re, e2_im, e3_re, e3_im;
    e0_re = e0_im = e1_re = e1_im = e2_re = e2_im = e3_re = e3_im = 0;

    int n_p;
    int n = 0;
    for (n_p = 0; n_p < N_p; n_p++) {
        uint32_t e = E[F * D / 2 * N_p * blockIdx.z + F * D / 2 * n_p + D / 2 * blockIdx.x + 32 * blockIdx.y + threadIdx.x];

        e0_re = float(e & 0xf);
        e0_im = float((e >> 4) & 0xf);
        e1_re = float((e >> 8) & 0xf);
        e1_im = float((e >> 12) & 0xf);
        e2_re = float((e >> 16) & 0xf);
        e2_im = float((e >> 20) & 0xf);
        e3_re = float((e >> 24) & 0xf);
        e3_im = float((e >> 28) & 0xf);

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


        // if n = N, write and reset s1, s2
        n++;
        if (n % N == 0) {
            S1[F * D / 2 * N * blockIdx.z + F * D / 2 * ((n - 1) / N) + D / 2 * blockIdx.x + 32 * blockIdx.y + threadIdx.x].x = s1_0;
            S1[F * D / 2 * N * blockIdx.z + F * D / 2 * ((n - 1) / N) + D / 2 * blockIdx.x + 32 * blockIdx.y + threadIdx.x].y = s1_1;
            S1[F * D / 2 * N * blockIdx.z + F * D / 2 * ((n - 1) / N) + D / 2 * blockIdx.x + 32 * blockIdx.y + threadIdx.x].z = s1_2;
            S1[F * D / 2 * N * blockIdx.z + F * D / 2 * ((n - 1) / N) + D / 2 * blockIdx.x + 32 * blockIdx.y + threadIdx.x].w = s1_3;

            S2[F * D / 2 * N * blockIdx.z + F * D / 2 * ((n - 1) / N) + D / 2 * blockIdx.x + 32 * blockIdx.y + threadIdx.x].x = s2_0;
            S2[F * D / 2 * N * blockIdx.z + F * D / 2 * ((n - 1) / N) + D / 2 * blockIdx.x + 32 * blockIdx.y + threadIdx.x].y = s2_1;
            S2[F * D / 2 * N * blockIdx.z + F * D / 2 * ((n - 1) / N) + D / 2 * blockIdx.x + 32 * blockIdx.y + threadIdx.x].z = s2_2;
            S2[F * D / 2 * N * blockIdx.z + F * D / 2 * ((n - 1) / N) + D / 2 * blockIdx.x + 32 * blockIdx.y + threadIdx.x].w = s2_3;

            s1_0 = s1_1 = s1_2 = s1_3 = 0;
            s2_0 = s2_1 = s2_2 = s2_3 = 0;
        }
    }

    // write s1', s2'
    S1_p[F * D / 2 * blockIdx.z + D / 2 * blockIdx.x + 32 * blockIdx.y + threadIdx.x].x = s1_p_0;
    S1_p[F * D / 2 * blockIdx.z + D / 2 * blockIdx.x + 32 * blockIdx.y + threadIdx.x].y = s1_p_1;
    S1_p[F * D / 2 * blockIdx.z + D / 2 * blockIdx.x + 32 * blockIdx.y + threadIdx.x].z = s1_p_2;
    S1_p[F * D / 2 * blockIdx.z + D / 2 * blockIdx.x + 32 * blockIdx.y + threadIdx.x].w = s1_p_3;

    S2_p[F * D / 2 * blockIdx.z + D / 2 * blockIdx.x + 32 * blockIdx.y + threadIdx.x].x = s2_p_0;
    S2_p[F * D / 2 * blockIdx.z + D / 2 * blockIdx.x + 32 * blockIdx.y + threadIdx.x].y = s2_p_1;
    S2_p[F * D / 2 * blockIdx.z + D / 2 * blockIdx.x + 32 * blockIdx.y + threadIdx.x].z = s2_p_2;
    S2_p[F * D / 2 * blockIdx.z + D / 2 * blockIdx.x + 32 * blockIdx.y + threadIdx.x].w = s2_p_3;
}


int main() {
    uint32_t *h_E, *d_E;
    float *h_S1, *h_S2, *h_S1_p, *h_S2_p;
    float4 *d_S1, *d_S2, *d_S1_p, *d_S2_p;
    const size_t N = 2; // 256;
    const size_t N_p = 4; // 128 * 256;
    const size_t D = 64; // 512; // or 64
    const size_t T = 8; // 100000; // not set
    const size_t F = 2; // 256; // not set

    h_E = (uint32_t*)malloc(sizeof(uint32_t) * D / 2 * F * T);
    h_S1 = (float*)malloc(4 * sizeof(float) * D / 2 * F * (T/N));
    h_S2 = (float*)malloc(4 * sizeof(float) * D / 2 * F * (T/N));
    h_S1_p = (float*)malloc(4 * sizeof(float) * D / 2 * F * (T/N_p));
    h_S2_p = (float*)malloc(4 * sizeof(float) * D / 2 * F * (T/N_p));


    for (int i = 0; i < D / 2 * F * T; i++) {
        uint32_t v = i << 28;
        v = v ^ ((i & 0xf) << 24);
        v = v ^ ((i & 0xf) << 20);
        v = v ^ ((i & 0xf) << 16);
        v = v ^ ((i & 0xf) << 12);
        v = v ^ ((i & 0xf) << 8);
        v = v ^ ((i & 0xf) << 4);
        v = v ^ (i & 0xf);

        h_E[i] = (uint32_t) v;

        // printf("%d %lu %lu \n", i, (unsigned long)v & 0xf, (unsigned long)v);
    }
    // printf("\n");

    // float e0_re, e0_im, e1_re, e1_im, e2_re, e2_im, e3_re, e3_im;
    // for (int l = 0; l < T; l++) {
    //     printf("t = %d\n", l);
    //     for (int r = 0; r < F; r++) {
    //         for (int c = 0; c < D / 2; c++) {
    //             uint32_t e = h_E[l * F * D / 2 + r * D / 2 + c];
    //             // e0_re = float(e & 0xf); 
    //             // e0_im = float((e >> 4) & 0xf);
    //             // e1_re = float((e >> 8) & 0xf);
    //             // e1_im = float((e >> 12) & 0xf);
    //             // e2_re = float((e >> 16) & 0xf);
    //             // e2_im = float((e >> 20) & 0xf);
    //             // e3_re = float((e >> 24) & 0xf);
    //             // e3_im = float((e >> 28) & 0xf);
    //             // printf("%f %f %f %f %f %f %f %f", e0_re, e1_im, e1_re, e1_im, e2_re, e2_im, e3_re, e3_im);
    //             // printf(" "); 
    //             printf("( %lu ) ", (unsigned long) e);
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");
    // }

    gpuErrchk(cudaMalloc((void**)&d_E, sizeof(uint32_t) * D / 2 * F * T));
    gpuErrchk(cudaMalloc((void**)&d_S1, sizeof(float4) * D / 2 * F * (T/N)));
    gpuErrchk(cudaMalloc((void**)&d_S2, sizeof(float4) * D / 2 * F * (T/N)));
    gpuErrchk(cudaMalloc((void**)&d_S1_p, sizeof(float4) * D / 2 * F * (T/N_p)));
    gpuErrchk(cudaMalloc((void**)&d_S2_p, sizeof(float4) * D / 2 * F * (T/N_p)));

    gpuErrchk(cudaMemcpy(d_E, h_E, sizeof(uint32_t) * D / 2 * F * T, cudaMemcpyHostToDevice));

    dim3 blocks(F, D / (32 * 2), T/N_p);
    dim3 threads(32);  // originally 2D/4. 2D bc dish and x- or y- polarisation pairs, 
                    // /4 bc 16 registers/thread, each holds 4 feeds. 16/4=4, one for each output array

    clock_t before = clock();

    downsample<<< blocks, threads >>>(d_E, d_S1, d_S2, d_S1_p, d_S2_p, N, N_p, D, T, F); 
    
    gpuErrchk(cudaDeviceSynchronize());

    double difference = (double)(clock() - before) / CLOCKS_PER_SEC;
    printf("Total time taken: %f \n", difference);

    gpuErrchk(cudaMemcpy(h_S1, d_S1, 4 * sizeof(float) * D / 2 * F * (T/N), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_S2, d_S2, 4 * sizeof(float) * D / 2 * F * (T/N), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_S1_p, d_S1_p, 4 * sizeof(float) * D / 2 * F * (T/N_p), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_S2_p, d_S2_p, 4 * sizeof(float) * D / 2 * F * (T/N_p), cudaMemcpyDeviceToHost));

    // for (int t = 0; t < T / N_p; t++) {
    //     for (int f = 0; f < F; f++) {
    //         for (int feed = 0; feed < 4 * D / 2; feed++) {
    //             printf("%f ", h_S1_p[t * F * D / 2 + f * D / 2 + feed];);
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");
    // }
}