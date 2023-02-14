// code source: https://solarianprogrammer.com/2012/05/31/matrix-multiplication-cuda-cublas-curand-thrust/

#include <iostream>
#include <cstdlib>
#include <curand.h> // for temp function
#include <cublas_v2.h>
#include <string.h>


// TEMPORARY FUNCTION
// Fill the array A(nr_rows_A, nr_cols_A) with random numbers on GPU
void GPU_fill_rand(float *A, int nr_rows_A, int nr_cols_A) {
	// Create a pseudo-random number generator
	curandGenerator_t prng;
	curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

	// Set the seed for the random number generator using the system clock
	curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());

	// Fill the array with random numbers on the device
	curandGenerateUniform(prng, A, nr_rows_A * nr_cols_A);
}


//Print matrix A(nr_rows_A, nr_cols_A) storage in column-major format
void print_matrix(const float *A, int nr_rows_A, int nr_cols_A, int nr_depth_A) {
	for (int k = 0; k < nr_depth_A; k++) {
		std::cout << "k = " << k << std::endl;
		for(int i = 0; i < nr_rows_A; ++i){
			for(int j = 0; j < nr_cols_A; ++j){
				std::cout << A[k * nr_rows_A * nr_cols_A + j * nr_rows_A + i] << " ";
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}
	std::cout << "----------" << std::endl;
}


int main( int argc, char *argv[] ) {
	// Allocate 3 arrays on CPU
	// num rows and cols
	int F, B, rho, T;
	
	F = 2;
	B = 5000;
	rho = 48 * 48;
	T = 256;
	if ( argc == 2 ) {
		if (strcmp(argv[1], "PF") == 0) 
		{
			B = 800;
			rho = 16 * 24;
			T = 256;
		} else if (strcmp(argv[1], "test") == 0) { // can delete later
			B = 5; 
			rho = 4;
			T = 2; 
		}
	}

	// Create a handle for CUBLAS
	cublasHandle_t handle;
	cublasCreate(&handle);
	
	float alpha = 1;
	float beta = 0;
	
	float *h_Iin = (float *)malloc(F * T * rho * sizeof(float));
	float *h_W = (float *)malloc(F * rho * B * sizeof(float));
	float *h_Iout = (float *)malloc(F * T * B * sizeof(float));

	// for testing, delete later
	if (argc == 2 and strcmp(argv[1], "test") == 0) {
		for (int j = 0; j < F; j++) {
			for (int i = j * T * rho; i < (j + 1) * T * rho; i++) { 
				h_Iin[i] = 0;
				if (i == 1 + j * T * rho or i == 2 + j * T * rho or i == 6 + j * T * rho) { h_Iin[i] = 1; }
			}
			for (int i = j * rho * B; i < (j + 1) * rho * B; i++) {
				h_W[i] = 0;
				if (i == 2 + j * B * rho or i == 5 + j * B * rho or i == 7 + j * B * rho or i == 14 + j * B * rho or i == 15 + j * B * rho or i == 18) { h_W[i] = 1; } 	
			}
		}
		std::cout << "Iin =" << std::endl;
		print_matrix(h_Iin, T, rho, F);
		std::cout << "W =" << std::endl;
		print_matrix(h_W, B, rho, F);

	}
		
	// Allocate 3 arrays on GPU
	float *d_Iin, *d_W, *d_Iout;
	cudaMalloc(&d_Iin,T * rho * sizeof(float));
	cudaMalloc(&d_W,rho * B * sizeof(float));
	cudaMalloc(&d_Iout,T * B * sizeof(float));

	for (int f = 0; f < F; f++) {
		if (argc == 2 and strcmp(argv[1], "test") != 0) { // do f times and copy into appropriate place in CPU arrays
			// temp until real data is passed in
			// Fill the arrays A and B on GPU with random numbers
			GPU_fill_rand(d_Iin, T, rho * F);
			GPU_fill_rand(d_W, rho, B * F);
			cudaMemcpy(h_Iin,d_Iin,T * rho * sizeof(float),cudaMemcpyDeviceToHost);
			cudaMemcpy(h_W,d_W,rho * B * sizeof(float),cudaMemcpyDeviceToHost);
		} else {
			// Copy data from CPU to GP
			cudaMemcpy(d_Iin,h_Iin + f * T * rho, T * rho * sizeof(float),cudaMemcpyHostToDevice);
			cudaMemcpy(d_W,h_W + f * rho * B, rho * B * sizeof(float),cudaMemcpyHostToDevice);

		}

		// Multiply A and B on GPU
		cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, T, B, rho, &alpha, d_Iin, T, d_W, B, &beta, d_Iout, T);

		// Copy the result on host memory
		cudaMemcpy(h_Iout + f * T * B, d_Iout,T * B * sizeof(float),cudaMemcpyDeviceToHost);

	}

	// Print Result 
	std::cout << "Iout =" << std::endl;
	print_matrix(h_Iout, T, B, F);
	
	// Destroy the handle
	cublasDestroy(handle);
	
	//Free GPU memory
	cudaFree(d_Iin);
	cudaFree(d_W);
	cudaFree(d_Iout);

	// Free CPU memory
	free(h_Iin);
	free(h_W);
	free(h_Iout);

	return 0;
}
