// code source: https://solarianprogrammer.com/2012/05/31/matrix-multiplication-cuda-cublas-curand-thrust/

#include <iostream>
#include <cstdlib>
#include <curand.h> // for temp function
#include <cublas_v2.h>
#include <string.h>
#include <assert.h>
#include <sys/time.h>
#include "fp16_conversion.h"


double gettime() {
  struct timeval tp;
  gettimeofday(&tp, nullptr);
  return tp.tv_sec + tp.tv_usec / 1.0e+6;
}


// TEMPORARY FUNCTION
// Fill the array A(nr_rows_A, nr_cols_A) with random numbers on CPU
void fill_ones(__half *A, int n_elements) {
	for(int i = 0; i < n_elements; i++) {
		__half num = 1.0;
		A[i] = num;
	}
}


//Print matrix A(nr_rows_A, nr_cols_A) storage in column-major format
void print_matrix(const __half *A, int nr_rows_A, int nr_cols_A, int nr_depth_A) {
	for (int k = 0; k < nr_depth_A; k++) {
		std::cout << "k = " << k << std::endl;
		for(int i = 0; i < nr_rows_A; ++i){
			for(int j = 0; j < nr_cols_A; ++j){
				std::cout << (float) A[k * nr_rows_A * nr_cols_A + j * nr_rows_A + i];
				std::cout << " ";
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}
	std::cout << "----------" << std::endl;
}


int main( int argc, char *argv[] ) {
	size_t F, B, rho, T;
	
	F = 256;
	B = 5000;
	rho = 48 * 48;
	T = 256;
	if ( argc == 2 ) {
		if (strcmp(argv[1], "PF") == 0) 
		{
			B = 800;
			rho = 16 * 24;
			T = 256;
			B = 5; 
		} else if (strcmp(argv[1], "test") == 0 or strcmp(argv[1], "testones") == 0) { // can delete later
			rho = 4;
			T = 2;
		    F = 2;
			B = 3;
		}
	}

	// Create a handle for CUBLAS
	cublasHandle_t handle;
	cublasCreate(&handle);
	
	__half alpha = 1;
	__half beta = 0;

	__half *h_Iin = (__half *)malloc(F * T * rho * sizeof(__half));
	__half *h_W = (__half *)malloc(F * rho * B * sizeof(__half));
	__half *h_Iout = (__half *)malloc(F * T * B * sizeof(__half));
	if (h_Iin == NULL) {
		std::cout << "Iin Malloc failed!" << std::endl;
		return;
	} else if (h_W == NULL) {
		std::cout << "W Malloc failed!" << std::endl;
		return;
	} else if (h_Iout == NULL) {
		std::cout << "Iout Malloc failed!" << std::endl;
		return;
	}

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
		
	} else {
		// temp fill with fake data
		fill_ones(h_Iin, T * rho * F);
		fill_ones(h_W, rho * B * F);
	}
 	
	if (argc == 2 and (strcmp(argv[1], "test") == 0 or strcmp(argv[1], "testones") == 0)) { // can delete later
		std::cout << "Iin =" << std::endl;
		print_matrix(h_Iin, T, rho, F);
		std::cout << "W =" << std::endl;
		print_matrix(h_W, B, rho, F);
	}

	// Allocate 3 arrays on GPU
	__half *d_Iin, *d_W, *d_Iout;
	int stat1 = cudaMalloc(&d_Iin, T * rho * sizeof(__half));
	int stat2 = cudaMalloc(&d_W, rho * B * sizeof(__half));
	int stat3 = cudaMalloc(&d_Iout, T * B * sizeof(__half));
	if (stat1 != 0 or stat2 != 0 or stat3 != 0) {
		std::cout << "cudaMalloc failed!" << std::endl;
		return;
	}


	// start timer
	double time0 = gettime();

	for (int f = 0; f < F; f++) {	
		// Copy data from CPU to GP
		cudaMemcpy(d_Iin,h_Iin + f * T * rho, T * rho * sizeof(__half),cudaMemcpyHostToDevice);
		cudaMemcpy(d_W,h_W + f * rho * B, rho * B * sizeof(__half),cudaMemcpyHostToDevice);

		// Multiply A and B^T on GPU
		cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, T, B, rho, &alpha, d_Iin, T, d_W, B, &beta, d_Iout, T);

		// Copy the result on host memory
		cudaMemcpy(h_Iout + f * T * B, d_Iout,T * B * sizeof(__half),cudaMemcpyDeviceToHost);

	}

	// end timer
	double time1 = gettime();
	std::cout << "Operations finished in " << time1 - time0 << "s" << std::endl;

	if (argc == 2 and (strcmp(argv[1], "test") == 0 or strcmp(argv[1], "testones") == 0)) { // can delete later
		// Print Result 
		std::cout << "Iout =" << std::endl;
		print_matrix(h_Iout, T, B, F);
	}

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
