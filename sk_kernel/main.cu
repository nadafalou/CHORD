#include "sk_kernel.cuh"

int main() {
    uint32_t *h_E, *d_E;
    float *h_S1, *h_S2, *h_S1_p, *h_S2_p;
    float4 *d_S1, *d_S2, *d_S1_p, *d_S2_p;
    // Commented numbers are the real ones, current are for testing
    const size_t N = 256;
    const size_t N_p = 128 * 256;
    const size_t D = 512; // or 64
    const size_t T = 98304; // not set
    const size_t F = 256; // not set

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
    dim3 blocks(F,  2 * D / (32 * 4), T/N_p); // Each block contains 32 threads, each with 1 freq channel, 
                                        // 4 feeds (packed into uint32) and N' time samples
                                        // (the 2 is for polarisation, since each feed is a dish-polarisation pair)
    dim3 threads(32);

    clock_t before = clock();

    for (int i = 0; i < 10; i++) {
        downsample<<< blocks, threads >>>(d_E, d_S1, d_S2, d_S1_p, d_S2_p, N, N_p, D, T, F); 
    }
    
    gpuErrchk(cudaDeviceSynchronize());

    double difference = (double)(clock() - before) / CLOCKS_PER_SEC;
    float bw = ((D * 2 * F * T) + 2 * (D * 2 * F * T/N_p) + 2 * (D * 2 * F * T/N)) / 1000000000 / difference;
    printf("Total time taken: %f s \n Runtime bandwidth: %f GB/s\n", difference / 10, bw * 10);

    // Copy output arrays
    // gpuErrchk(cudaMemcpy(h_S1, d_S1, 4 * sizeof(float) * D / 2 * F * (T/N), cudaMemcpyDeviceToHost));
    // gpuErrchk(cudaMemcpy(h_S2, d_S2, 4 * sizeof(float) * D / 2 * F * (T/N), cudaMemcpyDeviceToHost));
    // gpuErrchk(cudaMemcpy(h_S1_p, d_S1_p, 4 * sizeof(float) * D / 2 * F * (T/N_p), cudaMemcpyDeviceToHost));
    // gpuErrchk(cudaMemcpy(h_S2_p, d_S2_p, 4 * sizeof(float) * D / 2 * F * (T/N_p), cudaMemcpyDeviceToHost));
}