#include "sk_kernel.cuh"
#include "mask.cuh"

void run_downsample() {
    uint32_t *h_E, *d_E;
    float *h_S1, *h_S2, *h_S1_p, *h_S2_p;
    float4 *d_S1, *d_S2, *d_S1_p, *d_S2_p;
    // Commented numbers are the real ones, current are for testing
    const size_t N = 256;
    const size_t N_p = 128 * 256;
    const size_t D = 512; // 512 or 64
    const size_t T = 98304; // not set
    const size_t F = 256; // not set
    const int num_runs = 10;

    h_E = (uint32_t*)malloc(sizeof(uint32_t) * D / 2 * F * T);
    h_S1 = (float*)malloc(4 * sizeof(float) * D / 2 * F * (T/N));
    h_S2 = (float*)malloc(4 * sizeof(float) * D / 2 * F * (T/N));
    h_S1_p = (float*)malloc(4 * sizeof(float) * D / 2 * F * (T/N_p));
    h_S2_p = (float*)malloc(4 * sizeof(float) * D / 2 * F * (T/N_p));

    // generate data
    generate_random(h_E, D / 2 * F * T);

    gpuErrchk(cudaMalloc((void**)&d_E, sizeof(uint32_t) * D / 2 * F * T));
    gpuErrchk(cudaMalloc((void**)&d_S1, sizeof(float4) * D / 2 * F * (T/N)));
    gpuErrchk(cudaMalloc((void**)&d_S2, sizeof(float4) * D / 2 * F * (T/N)));
    gpuErrchk(cudaMalloc((void**)&d_S1_p, sizeof(float4) * D / 2 * F * (T/N_p)));
    gpuErrchk(cudaMalloc((void**)&d_S2_p, sizeof(float4) * D / 2 * F * (T/N_p)));

    // Copy input array
    gpuErrchk(cudaMemcpy(d_E, h_E, sizeof(uint32_t) * D / 2 * F * T, cudaMemcpyHostToDevice));

    // Block and thread dimensions
    // TODO safe to assume D = 64 or 512 ONLY?
    dim3 blocks(F, D == 64 ? D / (32 * 2) : D / (32 * 4 * 2), T/N_p); // Each block contains 1 or 4 warps, each 32 threads, 
                                        // each with 1 freq channel, 
                                        // 4 feeds (packed into uint32) and N' time samples
                                        // (the 2 is for polarisation, since each feed is a dish-polarisation pair)
    dim3 threads(D == 64 ? 32 : 32 * 4);  // originally 2D/4. 2D bc dish and x- or y- polarisation pairs, 
                    // /4 bc 16 registers/thread, each holds 4 feeds. 16/4=4, one for each output array


    clock_t before = clock();

    for (int i = 0; i < num_runs; i++) {
        downsample<<< blocks, threads >>>(d_E, d_S1, d_S2, d_S1_p, d_S2_p, N, N_p, D, T, F); 
    }
    
    gpuErrchk(cudaDeviceSynchronize());

    double difference = (double)(clock() - before) / CLOCKS_PER_SEC;
    float bw = ((D * 2 * F * T) + 2 * (D * 2 * F * T/N_p) + 2 * (D * 2 * F * T/N)) / 1000000000 / difference;
    printf("Total time taken: %f s \n Runtime bandwidth: %f GB/s\n", difference / num_runs, bw * num_runs);
}

void run_mask() {
    uint32_t *h_E, *d_E;
    uint32_t *h_R, *d_R, *h_W, *d_W;
    float *h_S1, *h_S2, *h_S1_p, *h_S2_p;
    float4 *d_S1, *d_S2, *d_S1_p, *d_S2_p;

    const size_t N = 256;
    const size_t N_p = 128 * 256;
    const size_t D = 512; // 512 or 64
    const size_t T = 98304; // not set
    const size_t T_bar = T / N;
    const size_t F = 256; // not set
    const float sigma = 5;
    const float N_good_min = 1;
    const float mu_min = 1;
    const int num_runs = 10;

    printf("Mallocing...\n");

    h_E = (uint32_t*)malloc(sizeof(uint32_t) * D / 2 * F * T);
    h_R = (uint32_t*)malloc(sizeof(uint32_t) * F * T_bar);
    h_W = (uint32_t*)malloc(sizeof(uint32_t) * D * 2);
    h_S1 = (float*)malloc(4 * sizeof(float) * D / 2 * F * (T/N));
    h_S2 = (float*)malloc(4 * sizeof(float) * D / 2 * F * (T/N));
    h_S1_p = (float*)malloc(4 * sizeof(float) * D / 2 * F * (T/N_p));
    h_S2_p = (float*)malloc(4 * sizeof(float) * D / 2 * F * (T/N_p));

    printf("Filling arrays with random data...\n");

    // generate data
    // Note: E generated using generate_random here bc generate_noise_array takes too long on large arrays
    generate_random(h_E, D / 2 * F * T);
    // generate_noise_array(h_E, D / 2 * F * T);
    generate_random_ones(h_W, D * 2);

    printf("cudaMallocing...\n");

    gpuErrchk(cudaMalloc((void**)&d_E, sizeof(uint32_t) * D / 2 * F * T));
    gpuErrchk(cudaMalloc((void**)&d_R, sizeof(uint32_t) * F * T_bar));
    gpuErrchk(cudaMalloc((void**)&d_W, sizeof(uint32_t) * D * 2));
    gpuErrchk(cudaMalloc((void**)&d_S1, sizeof(float4) * D / 2 * F * (T/N)));
    gpuErrchk(cudaMalloc((void**)&d_S2, sizeof(float4) * D / 2 * F * (T/N)));
    gpuErrchk(cudaMalloc((void**)&d_S1_p, sizeof(float4) * D / 2 * F * (T/N_p)));
    gpuErrchk(cudaMalloc((void**)&d_S2_p, sizeof(float4) * D / 2 * F * (T/N_p)));

    // temp for testing
    float *d_SK, *d_mean_SK, *d_var_SK, *h_SK, *h_mean_SK, *h_var_SK; // temp for testing
    h_SK = (float*)malloc(sizeof(float) * F * T/N);
    h_mean_SK = (float*)malloc(sizeof(float) * F * T/N);
    h_var_SK = (float*)malloc(sizeof(float) * F * T/N);
    gpuErrchk(cudaMalloc((void**)&d_SK, sizeof(float) * F * T/N));
    gpuErrchk(cudaMalloc((void**)&d_mean_SK, sizeof(float) * F * T/N));
    gpuErrchk(cudaMalloc((void**)&d_var_SK, sizeof(float) * F * T/N));

    printf("Copying E to device...\n");

    // Copy input array
    gpuErrchk(cudaMemcpy(d_E, h_E, sizeof(uint32_t) * D / 2 * F * T, cudaMemcpyHostToDevice));

    // Block and thread dimensions
    // TODO safe to assume D = 64 or 512 ONLY?
    dim3 blocks(F, D == 64 ? D / (32 * 2) : D / (32 * 4 * 2), T/N_p); // Each block contains 1 or 4 warps, each 32 threads, 
                                        // each with 1 freq channel, 
                                        // 4 feeds (packed into uint32) and N' time samples
                                        // (the 2 is for polarisation, since each feed is a dish-polarisation pair)
    dim3 threads(D == 64 ? 32 : 32 * 4);  // originally 2D/4. 2D bc dish and x- or y- polarisation pairs, 
                    // /4 bc 16 registers/thread, each holds 4 feeds. 16/4=4, one for each output array

    printf("Running downsample kernel...\n");

    clock_t before = clock();

    for (int i = 0; i < num_runs; i++) {
        downsample<<< blocks, threads >>>(d_E, d_S1, d_S2, d_S1_p, d_S2_p, N, N_p, D, T, F); 
    }
    
    gpuErrchk(cudaDeviceSynchronize());

    double difference = (double)(clock() - before) / CLOCKS_PER_SEC;
    // double bw = ((D * 2 * F * T) + 2 * (D * 2 * F * T/N_p) + 2 * (D * 2 * F * T/N)) / 1000000000 / difference; // E + S1 + S2 + S1' + S2'
    double bw = (2 * (D * 2 * F * T/N_p) + 2 * (D * 2 * F * T/N)) / 1000000000 / difference; // S1 + S2 + S1' + S2'
    printf("**Downsample Kernel**\n");
    printf("Total time taken: %f s \n Runtime bandwidth: %f GB/s\n", difference / num_runs, bw * num_runs);
    
    // Don't need these for masking kernel
    cudaFree(d_S1_p);
    cudaFree(d_S2_p);
    cudaFree(d_E);
    free(h_S1_p);
    free(h_S2_p);
    free(h_E);

    printf("Copying W to device...\n");

    gpuErrchk(cudaMemcpy(d_W, h_W, sizeof(uint32_t) * D * 2, cudaMemcpyHostToDevice));
    
    // define num blocks and threads
    dim3 blocks_mask(F, T_bar / 32);
    dim3 threads_mask(32 * 32); // 32 warps, each 32 threads (one coarse time index t_bar computed on each warp)

    printf("Rinning masking kernel...\n");

    // time and run parallel solution
    clock_t before_mask = clock();
    
    for (int i = 0; i < num_runs; i++) {
        mask<<< blocks_mask, threads_mask >>>(d_R, d_W, (float*) d_S1, (float*) d_S2, N, D, T_bar, F, mu_min, N_good_min, sigma, d_SK, d_mean_SK, d_var_SK); 
    }
   
    gpuErrchk(cudaDeviceSynchronize());

    double mask_difference = (double)(clock() - before_mask) / CLOCKS_PER_SEC;
    // double mask_bw = (double) ((F * T_bar) + (D * 2) + 2 * (D * 2 * F * T_bar)) / 1000000000 / mask_difference; // R + W + S1 + S2
    double mask_bw = (double) ((D * 2) + 2 * (D * 2 * F * T_bar)) / 1000000000 / mask_difference; // W + S1 + S2
    printf("**Masking Kernel**\n");
    printf("Total time taken: %f s \n Runtime bandwidth: %f GB/s\n", mask_difference / num_runs, mask_bw * num_runs);
}

int main() {
    // run_downsample();
    run_mask();
}