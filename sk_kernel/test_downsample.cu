#include <stdbool.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "sk_kernel.cuh" 


bool arrays_equal(float4 *arr1, float4 *arr2, size_t size) {
    for (size_t i = 0; i < size; i++) {
        if (arr1[i].x != arr2[i].x & arr1[i].y != arr2[i].y & arr1[i].z != arr2[i].z & arr1[i].w != arr2[i].w) {
            printf("at index %lu: %f != %f, %f != %f, %f != %f, %f != %f \n", 
                i, arr1[i].x, arr2[i].x, arr1[i].y, arr2[i].y, arr1[i].z, arr2[i].z, arr1[i].w, arr2[i].w);
            return false;
        }
    }
    return true;
}

float h_square(float num) {
    return num * num;
}

float h_cmplx_square(float real, float imaginary) {
    return h_square(real) + h_square(imaginary);
}

float h_cmplx_tesseract(float real, float imaginary) { 
    return h_square(h_cmplx_square(real, imaginary));
}

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

void naive_downsample(uint32_t *E, float4 *S1, float4 *S2, float4 *S1_p, float4 *S2_p, size_t N, size_t N_p, size_t D, size_t T, size_t F) {
    float s1_0, s1_1, s1_2, s1_3, s2_0, s2_1, s2_2, s2_3;
    s1_0 = s1_1 = s1_2 = s1_3 = s2_0 = s2_1 = s2_2 = s2_3 = 0;
    float s1_p_0, s1_p_1, s1_p_2, s1_p_3, s2_p_0, s2_p_1, s2_p_2, s2_p_3;
    s1_p_0 = s1_p_1 = s1_p_2 = s1_p_3 = s2_p_0 = s2_p_1 = s2_p_2 = s2_p_3 = 0;
    float e0_re, e0_im, e1_re, e1_im, e2_re, e2_im, e3_re, e3_im;
    e0_re = e0_im = e1_re = e1_im = e2_re = e2_im = e3_re = e3_im = 0;

    for (int t_bar = 0; t_bar < T; t_bar = t_bar + N_p){
        for (int f = 0; f < F; f++) {
            for (int feed4 = 0; feed4 < D / 2; feed4++) {
                s1_p_0 = 0;
                s2_p_0 = 0;
                s1_p_1 = 0;
                s2_p_1 = 0;
                s1_p_2 = 0;
                s2_p_2 = 0;
                s1_p_3 = 0;
                s2_p_3 = 0;

                for (int t = t_bar; t < (t_bar + 1) * N_p; t++) {
                    uint32_t e = E[t * D / 2 * F + f * D / 2 + feed4];

                    e0_re = float(e & 0xf);
                    e0_im = float((e >> 4) & 0xf);
                    e1_re = float((e >> 8) & 0xf);
                    e1_im = float((e >> 12) & 0xf);
                    e2_re = float((e >> 16) & 0xf);
                    e2_im = float((e >> 20) & 0xf);
                    e3_re = float((e >> 24) & 0xf);
                    e3_im = float((e >> 28) & 0xf);

                    s1_p_0 += h_cmplx_square(e0_re, e0_im);
                    s1_p_1 += h_cmplx_square(e1_re, e1_im);
                    s1_p_2 += h_cmplx_square(e2_re, e2_im);
                    s1_p_3 += h_cmplx_square(e3_re, e3_im);

                    s2_p_0 += h_cmplx_tesseract(e0_re, e0_im);
                    s2_p_1 += h_cmplx_tesseract(e1_re, e1_im);
                    s2_p_2 += h_cmplx_tesseract(e2_re, e2_im);
                    s2_p_3 += h_cmplx_tesseract(e3_re, e3_im);
                }

                S1_p[t_bar/N_p * D / 2 * F + f * D / 2 + feed4].x = s1_p_0;
                S1_p[t_bar/N_p * D / 2 * F + f * D / 2 + feed4].y = s1_p_1;
                S1_p[t_bar/N_p * D / 2 * F + f * D / 2 + feed4].z = s1_p_2;
                S1_p[t_bar/N_p * D / 2 * F + f * D / 2 + feed4].w = s1_p_3;

                S2_p[t_bar/N_p * D / 2 * F + f * D / 2 + feed4].x = s2_p_0;
                S2_p[t_bar/N_p * D / 2 * F + f * D / 2 + feed4].y = s2_p_1;
                S2_p[t_bar/N_p * D / 2 * F + f * D / 2 + feed4].z = s2_p_2;
                S2_p[t_bar/N_p * D / 2 * F + f * D / 2 + feed4].w = s2_p_3;

            }
        }
    }

    for (int t_bar = 0; t_bar < T; t_bar = t_bar + N){
        for (int f = 0; f < F; f++) {
            for (int feed4 = 0; feed4 < D / 2; feed4++) {
                s1_0 = 0;
                s2_0 = 0;
                s1_1 = 0;
                s2_1 = 0;
                s1_2 = 0;
                s2_2 = 0;
                s1_3 = 0;
                s2_3 = 0;
                
                for (int t = t_bar; t < (t_bar + 1) * N; t++) {
                    uint32_t e = E[t * D / 2 * F + f * D / 2 + feed4];

                    e0_re = float(e & 0xf);
                    e0_im = float((e >> 4) & 0xf);
                    e1_re = float((e >> 8) & 0xf);
                    e1_im = float((e >> 12) & 0xf);
                    e2_re = float((e >> 16) & 0xf);
                    e2_im = float((e >> 20) & 0xf);
                    e3_re = float((e >> 24) & 0xf);
                    e3_im = float((e >> 28) & 0xf);

                    if ((t_bar == 2 | t_bar == 3) && f == 0 && feed4 == 0) {
                        printf("%f + i%f, %f + i%f, %f + i%f, %f + i%f \n", e0_re, e0_im, e1_re, e1_im, e2_re, e2_im, e3_re, e3_im);;
                    }

                    s1_0 += h_cmplx_square(e0_re, e0_im);
                    s1_1 += h_cmplx_square(e1_re, e1_im);
                    s1_2 += h_cmplx_square(e2_re, e2_im);
                    s1_3 += h_cmplx_square(e3_re, e3_im);

                    s2_0 += h_cmplx_tesseract(e0_re, e0_im);
                    s2_1 += h_cmplx_tesseract(e1_re, e1_im);
                    s2_2 += h_cmplx_tesseract(e2_re, e2_im);
                    s2_3 += h_cmplx_tesseract(e3_re, e3_im);
                }

                S1[t_bar/N * D / 2 * F + f * D / 2 + feed4].x = s1_0;
                S1[t_bar/N * D / 2 * F + f * D / 2 + feed4].y = s1_1;
                S1[t_bar/N * D / 2 * F + f * D / 2 + feed4].z = s1_2;
                S1[t_bar/N * D / 2 * F + f * D / 2 + feed4].w = s1_3;

                S2[t_bar/N * D / 2 * F + f * D / 2 + feed4].x = s2_0;
                S2[t_bar/N * D / 2 * F + f * D / 2 + feed4].y = s2_1;
                S2[t_bar/N * D / 2 * F + f * D / 2 + feed4].z = s2_2;
                S2[t_bar/N * D / 2 * F + f * D / 2 + feed4].w = s2_3;

                if ((t_bar == 2 | t_bar == 3) && f == 0 && feed4 == 0) {
                    printf("at index %lu:%f %f %f %f \n", t_bar/N * D / 2 * F + f * D / 2 + feed4, s1_0, s1_1, s1_2, s1_3);;
                }
            }
        }
    }
}


void test_downsample() {
    uint32_t *h_E, *d_E;
    float4 *h_S1, *h_S2, *h_S1_p, *h_S2_p;
    float4 *naive_S1, *naive_S2, *naive_S1_p, *naive_S2_p;
    float4 *d_S1, *d_S2, *d_S1_p, *d_S2_p;
    const size_t N = 2;
    const size_t N_p = 4;
    const size_t D = 64;
    const size_t T = 12;
    const size_t F = 3;

    h_E = (uint32_t*)malloc(sizeof(uint32_t) * D / 2 * F * T);
    h_S1 = (float4*)malloc(sizeof(float4) * D / 2 * F * (T/N));
    h_S2 = (float4*)malloc(sizeof(float4) * D / 2 * F * (T/N));
    h_S1_p = (float4*)malloc(sizeof(float4) * D / 2 * F * (T/N_p));
    h_S2_p = (float4*)malloc(sizeof(float4) * D / 2 * F * (T/N_p));

    naive_S1 = (float4*)malloc(sizeof(float4) * D / 2 * F * (T/N));
    naive_S2 = (float4*)malloc(sizeof(float4) * D / 2 * F * (T/N));
    naive_S1_p = (float4*)malloc(sizeof(float4) * D / 2 * F * (T/N_p));
    naive_S2_p = (float4*)malloc(sizeof(float4) * D / 2 * F * (T/N_p));

    gpuErrchk(cudaMalloc((void**)&d_E, sizeof(uint32_t) * D / 2 * F * T));
    gpuErrchk(cudaMalloc((void**)&d_S1, sizeof(float4) * D / 2 * F * (T/N)));
    gpuErrchk(cudaMalloc((void**)&d_S2, sizeof(float4) * D / 2 * F * (T/N)));
    gpuErrchk(cudaMalloc((void**)&d_S1_p, sizeof(float4) * D / 2 * F * (T/N_p)));
    gpuErrchk(cudaMalloc((void**)&d_S2_p, sizeof(float4) * D / 2 * F * (T/N_p)));

    dim3 blocks(F, D / (32 * 2), T/N_p);
    dim3 threads(32);  // originally 2D/4. 2D bc dish and x- or y- polarisation pairs, 
                    // /4 bc 16 registers/thread, each holds 4 feeds. 16/4=4, one for each output array


    generate_random(h_E, D / 2 * F * T);

    for (int t = 0; t < T; t++) {
        for (int f = 0; f < F; f++) {
            for (int feed4 = 0; feed4 < D / 2; feed4++) {
                uint32_t e = h_E[t * F * D / 2 + f * D / 2 + feed4];
                float e0_im = float((e >> 4) & 0xf);
                float e0_re = float(e & 0xf);
                float e1_re = float((e >> 8) & 0xf);
                float e1_im = float((e >> 12) & 0xf);
                float e2_re = float((e >> 16) & 0xf);
                float e2_im = float((e >> 20) & 0xf);
                float e3_re = float((e >> 24) & 0xf);
                float e3_im = float((e >> 28) & 0xf);

                printf("%f + i%f, %f + i%f, %f + i%f, %f + i%f, ", e0_re, e0_im, e1_re, e1_im, e2_re, e2_im, e3_re, e3_im);
            }
            printf("\n");
        }
        printf("\n");
    }

    naive_downsample(h_E, naive_S1, naive_S2, naive_S1_p, naive_S2_p, N, N_p, D, T, F);

    gpuErrchk(cudaMemcpy(d_E, h_E, sizeof(uint32_t) * D / 2 * F * T, cudaMemcpyHostToDevice));

    downsample<<< blocks, threads >>>(d_E, d_S1, d_S2, d_S1_p, d_S2_p, N, N_p, D, T, F); 
    
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpy(h_S1, d_S1, sizeof(float4) * D / 2 * F * (T/N), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_S2, d_S2, sizeof(float4) * D / 2 * F * (T/N), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_S1_p, d_S1_p, sizeof(float4) * D / 2 * F * (T/N_p), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_S2_p, d_S2_p, sizeof(float4) * D / 2 * F * (T/N_p), cudaMemcpyDeviceToHost));

    if (arrays_equal(h_S1, naive_S1, D / 2 * F * (T/N)) == 0) {
        printf("S1 does not match \n");
    }
    if (arrays_equal(h_S1_p, naive_S1_p, D / 2 * F * (T/N_p)) == 0) {
        printf("S1' does not match \n");
    }
    if (arrays_equal(h_S2, naive_S2, D / 2 * F * (T/N)) == 0) {
        printf("S2 does not match \n");
    }
    if (arrays_equal(h_S2_p, naive_S2_p, D / 2 * F * (T/N_p)) == 0) {
        printf("S2' does not match \n");
    }
}


int main() {
    test_downsample();
}
