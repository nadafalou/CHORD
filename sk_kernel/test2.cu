#include "sk_kernel.cuh"

#include <cstdlib>
#include <cassert>
#include <iostream>

using namespace std;


// Generate a random complex integer, such that real/imag parts are between -8 and 7 inclusive.
// We use the cuda datatype 'int2' to represent a complex integer.

static inline int2 rand_complex_int()
{
    int2 ret;
    int r = rand();
    ret.x = (r & 0xf) - 8;          // use lowest 4 bits for real part
    ret.y = ((r >> 4) & 0xf) - 8;   // use next lowest 4 bits for imag part
    return ret;
}


// Randomize an array of many complex integers (int2s).
// If 'nsparse' is specified, then we randomize a random subset of the entries.
// This is useful because it allows testing in large examples, where the computational
// cost of rand() would otherwise be an issue.

static void randomize_E(int2 *E, long nelts, long nsparse=0)
{
    if (nsparse == 0) {
	// If nsparse is not specified, we randomize all elements.
	for (long i = 0; i < nelts; i++)
	    E[i] = rand_complex_int();
    }
    else {
	// If nsparse is specified, we randomize 'nsparse' randomly chosen elements.
	memset(E, 0, nelts * sizeof(int2));
	for (long s = 0; s < nsparse; s++) {
	    long i = long(rand()) + (long(rand()) << 30);
	    E[i % nelts] = rand_complex_int();
	}
    }
}


// Reference, CPU-based version of the downsample kernel.
// The 'M' parameter is the total number of "spectator" indices: M = (2 * dishes * freqs).

static void cpu_downsample(const int2 *E, uint *S1, uint *S2, long N, long T, long M)
{
    assert((T % N) == 0);
    long Tout = T / N;

    memset(S1, 0, Tout * M * sizeof(uint));
    memset(S2, 0, Tout * M * sizeof(uint));
    
    for (long tout = 0; tout < Tout; tout++) {
	for (long m = 0; m < M; m++) {
	    for (long tin = tout*N; tin < (tout+1)*N; tin++) {
		int2 e = E[tin*M + m];
		int sq = e.x*e.x + e.y*e.y;
		
		S1[tout*M + m] += sq;
		S2[tout*M + m] += sq*sq;
	    }
	}
    }
}


static bool compare_S_arrays(const uint *cpu_arr, const uint *gpu_arr, long Tds, long M, const char *arr_name)
{
    for (long t = 0; t < Tds; t++) {
	for (long m = 0; m < M; m++) {
	    uint cpu = cpu_arr[t*M+m];
	    uint gpu = gpu_arr[t*M+m];
	    
	    if (cpu == gpu)
		continue;
	    
	    cout << "    " << arr_name << " first mismatch at t=" << t
		 << ", m=" << m << " cpu=" << cpu << ", gpu=" << gpu << endl;

	    return false;
	}
    }

    cout << "    " << arr_name << ": looks good" << endl;
    return true;
}


// If 'nsparse' is specified, then we randomize a random subset of the E-array
// (the rest of the E-array is zeroed). This is useful because it allows testing
// in large examples, where the computational cost of rand() would otherwise be an issue.

static void test2_downsample(long N, long N_p, long D, long T, long F, long nsparse=0)
{
    cout << "test2_downsample(N=" << N << ", N_p=" << N_p << ", D=" << D
	 << ", T=" << T << ", F=" << F << ", nsparse=" << nsparse << ")" << endl;
    
    assert((N_p % N) == 0);
    assert((T % N_p) == 0);
    
    // M = total number of "spectator" indices (frequencies, dishes, polarizations).
    long M = 2*F*D;
    
    // Allocate arrays for CPU kernel.
    // We represent the E-field as int32+32 (not int4+4), to avoid making the CPU and GPU
    // kernels too similar. We represent an int32+32 using the cuda datatype 'int2'.
    
    int2 *cpu_E = (int2 *) malloc(T * M * sizeof(int2));           // shape (T, M)
    uint *cpu_S1 = (uint *) malloc((T/N) * M * sizeof(uint));      // shape (T/N, M)
    uint *cpu_S2 = (uint *) malloc((T/N) * M * sizeof(uint));      // shape (T/N, M)
    uint *cpu_S1_p = (uint *) malloc((T/N_p) * M * sizeof(uint));  // shape (T/N_p, M)
    uint *cpu_S2_p = (uint *) malloc((T/N_p) * M * sizeof(uint));  // shape (T/N_p, M)
    
    // Generate a random E-array, and downsample it on the CPU (for comparison with GPU kernel).
    randomize_E(cpu_E, T*M, nsparse);
    // memset(cpu_E, 0, T * M * sizeof(int2));
    // cpu_E[0].x = -1;
    cpu_downsample(cpu_E, cpu_S1, cpu_S2, N, T, M);
    cpu_downsample(cpu_E, cpu_S1_p, cpu_S2_p, N_p, T, M);

    // Allocate device-side arrays for GPU kernel.
    
    uint32_t *d_E;
    uint4 *d_S1, *d_S2, *d_S1_p, *d_S2_p;
    gpuErrchk(cudaMalloc(&d_E, T * M));
    gpuErrchk(cudaMalloc(&d_S1, (T/N) * M * sizeof(uint)));
    gpuErrchk(cudaMalloc(&d_S2, (T/N) * M * sizeof(uint)));
    gpuErrchk(cudaMalloc(&d_S1_p, (T/N_p) * M * sizeof(uint)));
    gpuErrchk(cudaMalloc(&d_S2_p, (T/N_p) * M * sizeof(uint)));
    
    // Now copy the E-array from CPU to GPU.
    // We need to convert int32+32 -> int4+4 in this step, since the GPU kernel expects int4+4.
    // We represent int4+4 as (uint8_t) in this step.
    
    uint8_t *h_E = (uint8_t *) malloc(T * M);
    
    for (long i = 0; i < T*M; i++) {
	int re = cpu_E[i].x;
	int im = cpu_E[i].y;
	h_E[i] = (re & 0xf) | ((im & 0xf) << 4);  // int32+32 -> int4+4
    }

    gpuErrchk(cudaMemcpy(d_E, h_E, T * M, cudaMemcpyHostToDevice));

    // Run the GPU kernel.

    launch_downsample_kernel(d_E, d_S1, d_S2, d_S1_p, d_S2_p, N, N_p, D, T, F);
	
    // Copy output arrays from GPU to CPU.
    
    uint *gpu_S1 = (uint *) malloc((T/N) * M * sizeof(uint));      // shape (T/N, M)
    uint *gpu_S2 = (uint *) malloc((T/N) * M * sizeof(uint));      // shape (T/N, M)
    uint *gpu_S1_p = (uint *) malloc((T/N_p) * M * sizeof(uint));  // shape (T/N_p, M)
    uint *gpu_S2_p = (uint *) malloc((T/N_p) * M * sizeof(uint));  // shape (T/N_p, M)

    gpuErrchk(cudaMemcpy(gpu_S1, d_S1, (T/N) * M * sizeof(uint), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(gpu_S2, d_S2, (T/N) * M * sizeof(uint), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(gpu_S1_p, d_S1_p, (T/N_p) * M * sizeof(uint), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(gpu_S2_p, d_S2_p, (T/N_p) * M * sizeof(uint), cudaMemcpyDeviceToHost));

    // Compare outputs of CPU and GPU kernels
    
    bool pass = true;
    if (!compare_S_arrays(cpu_S1, gpu_S1, T/N, M, "S1"))
	pass = false;
    if (!compare_S_arrays(cpu_S2, gpu_S2, T/N, M, "S2"))
	pass = false;
    if (!compare_S_arrays(cpu_S1_p, gpu_S1_p, T/N_p, M, "S1_p"))
	pass = false;
    if (!compare_S_arrays(cpu_S2_p, gpu_S2_p, T/N_p, M, "S2_p"))
	pass = false;

    if (!pass)
	exit(1);
}


int main(int argc, char **argv)
{
    srand(time(NULL));

    // Usage: test2_downsample(N, N_p, D, T, F, nsparse=0)

    // Subscale cases with dense E-arrays
    test2_downsample(128, 64*128, 64, 2*64*128, 4);   // CHORD pathfinder with reduced (T,F)
    test2_downsample(256, 64*256, 256, 2*64*256, 2);  // HIRAX with reduced (T,F)
    test2_downsample(128, 64*128, 512, 2*64*128, 2);  // Full CHORD with reduced (T,F)
    test2_downsample(256, 64*256, 1024, 2*64*256, 2);  // CHIME with reduced (T,F)

    // Full-scale cases with sparse E-arrays
    test2_downsample(128, 64*128, 64, 5*64*128, 400, 100*1000);   // CHORD pathfinder
    test2_downsample(256, 64*256, 256, 6*64*256, 64, 100*1000);   // HIRAX
    test2_downsample(128, 64*128, 512, 5*64*128, 50, 100*1000);   // Full CHORD
    test2_downsample(256, 64*256, 1024, 6*64*256, 16, 100*1000);  // CHIME
    
    return 0;
}
