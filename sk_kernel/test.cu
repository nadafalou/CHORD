#include "sk_kernel.cuh" 
#include "mask.cuh"

bool float4_arrays_equal(float4 *arr1, float4 *arr2, size_t size) {
    for (size_t i = 0; i < size; i++) {
        if (arr1[i].x != arr2[i].x | arr1[i].y != arr2[i].y | arr1[i].z != arr2[i].z | arr1[i].w != arr2[i].w) {
            printf("first fail at index %lu: %f != %f, %f != %f, %f != %f, %f != %f \n", 
                i, arr1[i].x, arr2[i].x, arr1[i].y, arr2[i].y, arr1[i].z, arr2[i].z, arr1[i].w, arr2[i].w);
            return false;
        }
    }
    return true;
}

bool uint32_arrays_equal(uint32_t *arr1, uint32_t *arr2, size_t size) {
    for (size_t i = 0; i < size; i++) {
        if (arr1[i] != arr2[i]) {
            printf("first fail at index %lu: %lu != %lu \n", i, (unsigned long) arr1[i], (unsigned long) arr2[i]);
            return false;
        }
    }
    return true;
}

bool float_arrays_equal(float *arr1, float *arr2, size_t size) {
    for (size_t i = 0; i < size; i++) {
        if (0.7 > arr1[i] / arr2[i] | arr1[i] / arr2[i] > 1.3) {
            printf("first fail at index %lu: %f != %f \n", i, arr1[i], arr2[i]);
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

                for (int t = t_bar; t < t_bar + N_p; t++) {
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
                
                for (int t = t_bar; t < t_bar + N; t++) {
                    uint32_t e = E[t * D / 2 * F + f * D / 2 + feed4];

                    e0_re = float(e & 0xf);
                    e0_im = float((e >> 4) & 0xf);
                    e1_re = float((e >> 8) & 0xf);
                    e1_im = float((e >> 12) & 0xf);
                    e2_re = float((e >> 16) & 0xf);
                    e2_im = float((e >> 20) & 0xf);
                    e3_re = float((e >> 24) & 0xf);
                    e3_im = float((e >> 28) & 0xf);

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
            }
        }
    }
}


float M_func(float mu) {
    return 0;
}

float V_func(float mu) {
    return 1;
}


void naive_mask(uint32_t *R, uint32_t *W, float4 *S1, float4 *S2, size_t N, size_t D, size_t T_bar, size_t F, float mu_min, float N_good_min, float sigma, float *SK, float *mean_SK, float *var_SK) {
    float N_good = 0;
    for (int w = 0; w < D * 2; w++) {
        N_good += (int) W[w];
    }
    printf("N_good = %f \n", N_good);

    if (N_good < N_good_min) {
        for (int r = 0; r < F * T_bar / 32; r++) {
            R[r] = 0;
        }
        printf("NOT ENOUGH N_GOOD \n");
        return;
    }

    float mu[D * 2 * F * T_bar];
    float S2_tilde[D * 2 * F * T_bar];
    for (int s = 0; s < D / 2 * F * T_bar; s++) {
        mu[s * 4 + 0] = S1[s].x / (float) N;
        mu[s * 4 + 1] = S1[s].y / (float) N;
        mu[s * 4 + 2] = S1[s].z / (float) N;
        mu[s * 4 + 3] = S1[s].w / (float) N;

        if (mu[s * 4 + 0] < mu_min) { S2_tilde[s * 4 + 0] = 0; } 
        else { S2_tilde[s * 4 + 0] = S2[s].x / (mu[s * 4 + 0] * mu[s * 4 + 0]); }
        if (mu[s * 4 + 1] < mu_min) { S2_tilde[s * 4 + 1] = 0; } 
        else { S2_tilde[s * 4 + 1] = S2[s].y / (mu[s * 4 + 1] * mu[s * 4 + 1]); }
        if (mu[s * 4 + 2] < mu_min) { S2_tilde[s * 4 + 2] = 0; } 
        else { S2_tilde[s * 4 + 2] = S2[s].z / (mu[s * 4 + 2] * mu[s * 4 + 2]); }
        if (mu[s * 4 + 3] < mu_min) { S2_tilde[s * 4 + 3] = 0; } 
        else { S2_tilde[s * 4 + 3] = S2[s].w / (mu[s * 4 + 3] * mu[s * 4 + 3]); }
    }
    printf("S1[0]=%f, mu[0]=%f, S2[0]=%f, S2_tilde[0]=%f\n", S1[0].x, mu[0], S2[0].x, S2_tilde[0]); 

    float sum;
    float mean_sum;
    float var_sum;
    
    float frac = (1 / N_good) * (((float) N + 1) / (float) (N - 1));
    float mean_frac = 1 + 1 / N_good;
    float var_frac = 4 / (h_square(N_good) * (float) N);

    // these can be single floats since values are used to compute R in the same loop
    float sk; 
    float mean_sk;
    float var_sk;

    for (int f = 0; f < F; f++) {    
        for (int t = 0; t < T_bar; t++) {
            sum = 0;
            mean_sum = 0;
            var_sum = 0;
            for (int pd = 0; pd < D * 2; pd++) {
                if (f == 0 & t == 0){ printf("sum=%f, w=%f, mu=%f, s2_tilde=%f, new=%f\n", sum, (float) W[pd], mu[t * F * D * 2 + f * D * 2 + pd], S2_tilde[t * F * D * 2 + f * D * 2 + pd], (float) W[pd] * ((S2_tilde[t * F * D * 2 + f * D * 2 + pd] / (float) N) - 1.)); }
                if (mu[t * F * D * 2 + f * D * 2 + pd] >= mu_min){
                    sum += (float) W[pd] * ((S2_tilde[t * F * D * 2 + f * D * 2 + pd] / (float) N) - 1.);
                    mean_sum += (float) W[pd] * M_func(mu[t * F * D * 2 + f * D * 2 + pd]);
                    var_sum += (float) W[pd] * V_func(mu[t * F * D * 2 + f * D * 2 + pd]);
                }
                // if (f == 0 & t == 0 & pd == 0){ printf("s2_tilde[0]=%f\n", S2_tilde[t * F * D * 2 + f * D * 2 + pd]); }
                // if (f == 0 & t == 0) { printf("w=%lu, sum=%f, mean_sum=%f, var_sum=%f\n", (unsigned long) W[pd], sum, mean_sum, var_sum); }
            }
            // printf("f=%d, t=%d, sum=%f, mean_sum=%f, var_sum=%f\n", f, t, sum, mean_sum, var_sum);
            sk = frac * sum;
            mean_sk = mean_frac * mean_sum;
            var_sk = var_frac * var_sum;

            if (f == 0 & t == 0) { 
            printf("at f=t=0, frac=%f, sum=%f, sk=%f, mean_sk=%f, var_sk=%f\n", frac, sum, sk, mean_sk, var_sk);
            }

            // R[f * T_bar + t] = ((abs(sk - mean_sk) <= (sigma * sqrt(var_sk))) ? true: false);
            if (abs(sk - mean_sk) <= (sigma * sqrt(var_sk))) { R[f * T_bar + t] = true; }
            else { R[f * T_bar + t] = false; }

            // temp
            SK[f * T_bar + t] = sk;
            mean_SK[f * T_bar + t] = mean_sk;
            var_SK[f * T_bar + t] = var_sk;
        }
    }
}


void test_downsample() {
    uint32_t *h_E, *d_E;
    float4 *h_S1, *h_S2, *h_S1_p, *h_S2_p;
    float4 *naive_S1, *naive_S2, *naive_S1_p, *naive_S2_p;
    float4 *d_S1, *d_S2, *d_S1_p, *d_S2_p;
    const size_t N = 2;//256;
    const size_t N_p = 4;//256 * 128;
    const size_t D = 64; // 64 or 512
    const size_t T = 32;//98304;
    const size_t F = 1;//50;

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

    // TODO safe to assume D = 64 or 512 ONLY?
    dim3 blocks(F, D == 64 ? D / (32 * 2) : D / (32 * 4 * 2), T/N_p);
    dim3 threads(D == 64 ? 32 : 32 * 4);  // originally 2D/4. 2D bc dish and x- or y- polarisation pairs, 
                    // /4 bc 16 registers/thread, each holds 4 feeds. 16/4=4, one for each output array


    generate_random(h_E, D / 2 * F * T);

    // for (int t = 0; t < T; t++) {
    //     for (int f = 0; f < F; f++) {
    //         for (int feed4 = 0; feed4 < D / 2; feed4++) {
    //             uint32_t e = h_E[t * F * D / 2 + f * D / 2 + feed4];
    //             float e0_im = float((e >> 4) & 0xf);
    //             float e0_re = float(e & 0xf);
    //             float e1_re = float((e >> 8) & 0xf);
    //             float e1_im = float((e >> 12) & 0xf);
    //             float e2_re = float((e >> 16) & 0xf);
    //             float e2_im = float((e >> 20) & 0xf);
    //             float e3_re = float((e >> 24) & 0xf);
    //             float e3_im = float((e >> 28) & 0xf);

    //             printf("index %lu: %f + i%f, %f + i%f, %f + i%f, %f + i%f, ", t * F * D / 2 + f * D / 2 + feed4, e0_re, e0_im, e1_re, e1_im, e2_re, e2_im, e3_re, e3_im);
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");
    // }

    clock_t before_naive = clock();

    naive_downsample(h_E, naive_S1, naive_S2, naive_S1_p, naive_S2_p, N, N_p, D, T, F);

    double difference_naive = (double)(clock() - before_naive) / CLOCKS_PER_SEC;

    gpuErrchk(cudaMemcpy(d_E, h_E, sizeof(uint32_t) * D / 2 * F * T, cudaMemcpyHostToDevice));

    clock_t before = clock();

    downsample<<< blocks, threads >>>(d_E, d_S1, d_S2, d_S1_p, d_S2_p, N, N_p, D, T, F); 
    
    gpuErrchk(cudaDeviceSynchronize());

    double difference = (double)(clock() - before) / CLOCKS_PER_SEC;


    gpuErrchk(cudaMemcpy(h_S1, d_S1, sizeof(float4) * D / 2 * F * (T/N), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_S2, d_S2, sizeof(float4) * D / 2 * F * (T/N), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_S1_p, d_S1_p, sizeof(float4) * D / 2 * F * (T/N_p), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_S2_p, d_S2_p, sizeof(float4) * D / 2 * F * (T/N_p), cudaMemcpyDeviceToHost));

    bool match = true;
    if (float4_arrays_equal(h_S1, naive_S1, D / 2 * F * (T/N)) == 0) {
        printf("S1 does not match \n");
        match = false;
    }
    if (float4_arrays_equal(h_S1_p, naive_S1_p, D / 2 * F * (T/N_p)) == 0) {
        printf("S1' does not match \n");
        match = false;
    }
    if (float4_arrays_equal(h_S2, naive_S2, D / 2 * F * (T/N)) == 0) {
        printf("S2 does not match \n");
        match = false;
    }
    if (float4_arrays_equal(h_S2_p, naive_S2_p, D / 2 * F * (T/N_p)) == 0) {
        printf("S2' does not match \n");
        match = false;
    }

    printf("Naive runtime: %f \n", difference_naive);
    printf("Kernel runtime: %f \n", difference);
    printf("Solution match: %d \n", match);
}


void test_mask() {
    // declare everything
    uint32_t *h_E, *d_E;
    uint32_t *h_R, *d_R, *h_W, *d_W;
    uint32_t *naive_R;
    float4 *h_S1, *h_S2, *h_S1_p, *h_S2_p, *d_S1, *d_S2, *d_S1_p, *d_S2_p;
    const size_t N = 2;//2; 265;
    const size_t N_p = 4;//4; 256 * 128;
    const size_t D = 64; // 64 or 512, needs to be multiple of 64
    const size_t T = 64; //64; 98304;
    const size_t T_bar = T / N;
    const size_t F = 1;
    const float sigma = 5;
    float *naive_SK, *naive_mean_SK, *naive_var_SK, *d_SK, *d_mean_SK, *d_var_SK, *h_SK, *h_mean_SK, *h_var_SK; // temp for testing

    // malloc arrays on host
    h_E = (uint32_t*)malloc(sizeof(uint32_t) * D / 2 * F * T);
    h_R = (uint32_t*)malloc(sizeof(uint32_t) * F * T_bar);
    h_W = (uint32_t*)malloc(sizeof(uint32_t) * D * 2);
    h_S1 = (float4*)malloc(sizeof(float4) * D / 2 * F * T_bar);
    h_S2 = (float4*)malloc(sizeof(float4) * D / 2 * F * T_bar);
    h_S1_p = (float4*)malloc(sizeof(float4) * D / 2 * F * (T/N_p));
    h_S2_p = (float4*)malloc(sizeof(float4) * D / 2 * F * (T/N_p));

    naive_R = (uint32_t*)malloc(sizeof(uint32_t) * F * T_bar);

    // temp for testing
    naive_SK = (float*)malloc(sizeof(float) * F * T/N);
    naive_mean_SK = (float*)malloc(sizeof(float) * F * T/N);
    naive_var_SK = (float*)malloc(sizeof(float) * F * T/N);
    h_SK = (float*)malloc(sizeof(float) * F * T/N);
    h_mean_SK = (float*)malloc(sizeof(float) * F * T/N);
    h_var_SK = (float*)malloc(sizeof(float) * F * T/N);

    // malloc arrays on device
    gpuErrchk(cudaMalloc((void**)&d_E, sizeof(uint32_t) * D / 2 * F * T));
    gpuErrchk(cudaMalloc((void**)&d_R, sizeof(uint32_t) * F * T_bar));
    gpuErrchk(cudaMalloc((void**)&d_W, sizeof(uint32_t) * D * 2));
    gpuErrchk(cudaMalloc((void**)&d_S1, sizeof(float4) * D / 2 * F * T_bar));
    gpuErrchk(cudaMalloc((void**)&d_S2, sizeof(float4) * D / 2 * F * T_bar));
    gpuErrchk(cudaMalloc((void**)&d_S1_p, sizeof(float4) * D / 2 * F * T_bar));
    gpuErrchk(cudaMalloc((void**)&d_S2_p, sizeof(float4) * D / 2 * F * T_bar));

    // temp for testing
    gpuErrchk(cudaMalloc((void**)&d_SK, sizeof(float) * F * T/N));
    gpuErrchk(cudaMalloc((void**)&d_mean_SK, sizeof(float) * F * T/N));
    gpuErrchk(cudaMalloc((void**)&d_var_SK, sizeof(float) * F * T/N));

    // generate fake data 
    generate_random_ones(h_W, D * 2);
    generate_noise_array(h_E, D / 2 * F * T);
    // generate_random_float4(h_S1, D / 2 * F * T_bar);
    // generate_random_float4(h_S2, D / 2 * F * T_bar);

    dim3 down_blocks(F, D == 64 ? D / (32 * 2) : D / (32 * 4 * 2), T/N_p);
    dim3 down_threads(D == 64 ? 32 : 32 * 4);  // originally 2D/4. 2D bc dish and x- or y- polarisation pairs, 
                    // /4 bc 16 registers/thread, each holds 4 feeds. 16/4=4, one for each output array
    gpuErrchk(cudaMemcpy(d_E, h_E, sizeof(uint32_t) * D / 2 * F * T, cudaMemcpyHostToDevice));

    downsample<<< down_blocks, down_threads >>>(d_E, d_S1, d_S2, d_S1_p, d_S2_p, N, N_p, D, T, F); 

    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpy(h_S1, d_S1, sizeof(float4) * D / 2 * F * T_bar, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_S2, d_S2, sizeof(float4) * D / 2 * F * T_bar, cudaMemcpyDeviceToHost));

    // =================== MASK =====================

    // start the timer for naive solution
    clock_t before_naive = clock();

    // run naive solution
    naive_mask(naive_R, h_W, h_S1, h_S2, N, D, T_bar, F, 1, 1, sigma, naive_SK, naive_mean_SK, naive_var_SK);

    // print bunch of inputs and outputs TODO delete
    // printf("W \n");
    // for (int pd = 0; pd < D * 2; pd++) {
    //     printf("%d, ", h_W[pd]);
    // }
    // printf("\n");

    // printf("S1 \n");
    // for (int s = 0; s < D / 2; s++) { // s < D / 2 * F * T_bar
    //     printf("%f,%f,%f,%f,", h_S1[s].x, h_S1[s].y, h_S1[s].z, h_S1[s].w);
    // }
    // printf("\n");

    // printf("S2 \n");
    // for (int s = 0; s < D / 2; s++) { // s < D / 2 * F * T_bar
    //     printf("%f, %f, %f, %f, ", h_S2[s].x, h_S2[s].y, h_S2[s].z, h_S2[s].w);
    // }
    // printf("\n");

    // printf("S1[0] = %f, S2[0] = %f\n", h_S1[0].x, h_S2[0].x);

    // printf("R\n");
    // for (int f = 0; f < F; f++) {
    //     for (int t = 0; t < T_bar; t++) {
    //         printf("%d ", naive_R[f * T_bar + t]);
    //     }
    //     printf("\n");
    // }

    // end naive timer
    double difference_naive = (double)(clock() - before_naive) / CLOCKS_PER_SEC;

    // copy input host to device
    gpuErrchk(cudaMemcpy(d_W, h_W, sizeof(uint32_t) * D * 2, cudaMemcpyHostToDevice));
    // gpuErrchk(cudaMemcpy(d_S1, h_S1, sizeof(float4) * D / 2 * F * T_bar, cudaMemcpyHostToDevice));
    // gpuErrchk(cudaMemcpy(d_S2, h_S2, sizeof(float4) * D / 2 * F * T_bar, cudaMemcpyHostToDevice));

    // define num blocks and threads
    dim3 blocks(F, T_bar / 32);
    dim3 threads(32 * 32); // 32 warps, each 32 threads (one coarse time index t_bar computed on each warp)

    printf("blocks: %d, %d, %d; threads: %d, %d, %d \n", blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z);
    // time and run parallel solution
    clock_t before = clock();

    mask<<< blocks, threads >>>(d_R, d_W, (float*) d_S1, (float*) d_S2, N, D, T_bar, F, 1, 1, sigma, d_SK, d_mean_SK, d_var_SK); 
    printf("peek: %d\n", cudaPeekAtLastError());
   
    gpuErrchk(cudaDeviceSynchronize());
    printf("peek: %d\n", cudaPeekAtLastError());

    double difference = (double)(clock() - before) / CLOCKS_PER_SEC;

    // copy output device to host
    gpuErrchk(cudaMemcpy(h_R, d_R, sizeof(uint32_t) * F * T_bar, cudaMemcpyDeviceToHost));
    // test for testing
    gpuErrchk(cudaMemcpy(h_SK, d_SK, sizeof(float) * F * T_bar, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_mean_SK, d_mean_SK, sizeof(float) * F * T_bar, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_var_SK, d_var_SK, sizeof(float) * F * T_bar, cudaMemcpyDeviceToHost));

    // printf("GPU R\n");
    // for (int f = 0; f < F; f++) {
    //     for (int t = 0; t < T_bar; t++) {
    //         printf("%d ", h_R[f * T_bar + t]);
    //     }
    //     printf("\n");
    // }

    // check if solutions match
    bool match = true;
    if (uint32_arrays_equal(h_R, naive_R, F * T_bar) == 0) {
        printf("R does not match \n");
        match = false;
    }

    if (float_arrays_equal(h_SK, naive_SK, F * T_bar) == 0) {
        printf("SK does not match \n");
        match = false;
    }

    if (float_arrays_equal(h_mean_SK, naive_mean_SK, F * T_bar) == 0) {
        printf("mean_SK does not match \n");
        match = false;
    }

    if (float_arrays_equal(h_var_SK, naive_var_SK, F * T_bar) == 0) {
        printf("var_SK does not match \n");
        match = false;
    }

    // print results
    printf("Naive runtime: %f \n", difference_naive);
    printf("Kernel runtime: %f \n", difference);
    printf("Solution match: %d \n", match);
}


int main() {
    test_mask();
}
