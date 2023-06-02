#include <stdio.h>
#include <time.h>

#define BLOCK_SIZE 32


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

__global__ void transpose(float *d_in, float *d_out, int N){

    __shared__ float tile[BLOCK_SIZE][BLOCK_SIZE + 1];
    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int y = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    // NOTE: don't need this anymore bc now threads are in a grid (x and y)
    // row and col num in tile (note: possible to do this bc 32 is a power of 2)
    // int w = threadIdx.x >> 5; // warp # ~ row
    // int l = threadIdx.x & 31; // thread line # ~ col

    tile[threadIdx.x][threadIdx.y] = d_in[y * N + x];

    __syncthreads();
    
    // x and y of tile transpose (new transposed location of whole tile)
    x = blockIdx.y * BLOCK_SIZE + threadIdx.x; 
	y = blockIdx.x * BLOCK_SIZE + threadIdx.y;

    d_out[y * N + x] = tile[threadIdx.y][threadIdx.x];
}

void naive_transpose(float *h_in, float *h_out, int N){
    for (int row = 0; row < N; row++){
        for (int col = 0; col < N; col++){
            h_out[col * N + row] = h_in[row * N + col];
        }
    }
}

int main() {
    float *h_in, *h_out, *d_in, *d_out;
    const size_t N = 32000; // N = numrows = numcols

    h_in = (float*)malloc(sizeof(float) * N * N);
    h_out = (float*)malloc(sizeof(float) * N * N);

    for (int i = 0; i < N * N; i++) {
        h_in[i] = (float) i;
    }

    gpuErrchk(cudaMalloc((void**)&d_in, sizeof(float) * N * N));
    gpuErrchk(cudaMalloc((void**)&d_out, sizeof(float) * N * N));
    gpuErrchk(cudaMemcpy(d_in, h_in, sizeof(float) * N * N, cudaMemcpyHostToDevice));

    dim3 blocks(N / BLOCK_SIZE, N / BLOCK_SIZE, 1); 
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE, 1);

    clock_t before = clock();

    transpose<<< blocks, threads >>>(d_in, d_out, N); 
    gpuErrchk(cudaDeviceSynchronize());

    double difference = (double)(clock() - before) / CLOCKS_PER_SEC;
    printf("Total time taken: %f \n", difference);

    gpuErrchk(cudaMemcpy(h_out, d_out, sizeof(float) * N * N, cudaMemcpyDeviceToHost));

    // cross-check result with naive CPU method
    float *h_out_naive;
    h_out_naive = (float*)malloc(sizeof(float) * N * N);
    naive_transpose(h_in, h_out_naive, N);
    int equal = 1;
    for (int i = 0; i < N * N; i++) {if (h_out[i] != h_out_naive[i]) {equal = 0; break;}}
    if (equal == 1){printf("Test passed\n");} else {printf("TEST FAILED\n");}

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);

    return 0;
}