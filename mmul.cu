// code source: https://solarianprogrammer.com/2012/05/31/matrix-multiplication-cuda-cublas-curand-thrust/

#include <iostream>
#include <cstdlib>
#include <curand.h> // for temp function
#include <cublas_v2.h>
#include <string.h>
#include <assert.h>
#include <sys/time.h>


double gettime() {
  struct timeval tp;
  gettimeofday(&tp, nullptr);
  return tp.tv_sec + tp.tv_usec / 1.0e+6;
}


// Fill the array A(nr_rows_A, nr_cols_A) with random numbers on CPU
void fill_ones(__half *A, int n_elements) {
	for(int i = 0; i < n_elements; i++) {
		__half num = 1.0;
		A[i] = num;
	}
}


void fill_manual(__half *h_Iin, __half *h_W, __half **h_Iin_b, __half **h_W_b, size_t F, size_t B, size_t rho, size_t T) {
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

}


//Print matrix A(nr_rows_A, nr_cols_A) storage in column-major format
void print_matrix(const __half *A, int nr_rows_A, int nr_cols_A, int nr_depth_A) {
	for (int f = 0; f < nr_depth_A; f++) {
		std::cout << "f = " << f << std::endl;
		for(int i = 0; i < nr_rows_A; ++i){
			for(int j = 0; j < nr_cols_A; ++j){
				std::cout << (float) A[f * nr_rows_A * nr_cols_A + j * nr_rows_A + i];
				std::cout << " ";
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}
	std::cout << "----------" << std::endl;
}


void print_matrix_b(__half **A, int nr_rows_A, int nr_cols_A, int nr_depth_A) {
	for (int f = 0; f < nr_depth_A; f++) {
		std::cout << "f = " << f << std::endl;
		for(int i = 0; i < nr_rows_A; ++i){
			for(int j = 0; j < nr_cols_A; ++j){
				std::cout << (float) A[f][j * nr_rows_A + i];
				std::cout << " ";
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}
	std::cout << "----------" << std::endl;
}


void allocate_memory(__half **h_Iin, __half **h_W, __half **h_Iout, __half ***h_Iin_b, __half ***h_W_b, __half ***h_Iout_b, __half **d_Iin, __half **d_W, __half **d_Iout, __half ***d_Iin_b, __half ***d_W_b, __half ***d_Iout_b, size_t F, size_t B, size_t rho, size_t T) {
	if ((*h_Iin = (__half *)malloc(F * T * rho * sizeof(__half))) == NULL) {
		std::cout << "Iin Malloc failed!" << std::endl;
		exit(-1);
	}
	if ((*h_W = (__half *)malloc(F * rho * B * sizeof(__half))) == NULL) {
		std::cout << "W Malloc failed!" << std::endl;
		exit(-1);
	}
	if ((*h_Iout = (__half *)malloc(F * T * B * sizeof(__half))) == NULL) {
		std::cout << "Iout Malloc failed!" << std::endl;
		exit(-1);
	}

	// batch mem
	if ((*h_Iin_b = (__half **)malloc(F * sizeof(__half*))) == NULL) {
		std::cout << "Iin_b Malloc failed!" << std::endl;
		exit(-1);
	}
	if ((*h_W_b = (__half **)malloc(F * sizeof(__half*))) == NULL) {
		std::cout << "W_b Malloc failed!" << std::endl;
		exit(-1);
	}
	if ((*h_Iout_b = (__half **)malloc(F * sizeof(__half*))) == NULL) {
		std::cout << "Iout_b Malloc failed!" << std::endl;
		exit(-1);
	}

	// Now on device
	if (cudaMalloc(d_Iin, F * T * rho * sizeof(__half)) != 0) {
		std::cout << "cudaMalloc Iin failed!" << std::endl;
		exit(-1);
	}
	if (cudaMalloc(d_W, F * rho * B * sizeof(__half)) != 0) {
		std::cout << "cudaMalloc W failed!" << std::endl;
		exit(-1);
	}
	if (cudaMalloc(d_Iout, F * T * B * sizeof(__half)) != 0) {
		std::cout << "cudaMalloc Iout failed!" << std::endl;
		exit(-1);
	}

	if (cudaMalloc(d_Iin_b, F * sizeof(__half *)) != 0) {
		std::cout << "cudaMalloc Iin_b failed!" << std::endl;
		exit(-1);
	}
	if (cudaMalloc(d_W_b, F * sizeof(__half *)) != 0) {
		std::cout << "cudaMalloc W_b failed!" << std::endl;
		exit(-1);
	}
	if (cudaMalloc(d_Iout_b, F * sizeof(__half *)) != 0) {
		std::cout << "cudaMalloc Iout_b failed!" << std::endl;
		exit(-1);
	}

}


void HtoD_copy(__half **h_Iin, __half **h_W, __half ***h_Iin_b, __half ***h_W_b,  __half ***h_Iout_b, __half **d_Iin, __half **d_W, __half ***d_Iin_b, __half ***d_W_b, __half ***d_Iout_b, size_t F, size_t B, size_t rho, size_t T) {
	// TODO figure out if I need to pass in pointer to arrays or not, and if the args passed into cudaMemcpy are correct (dereference?)
	// Copy data from CPU to GP
	if (cudaMemcpy(*d_Iin, *h_Iin, F * T * rho * sizeof(__half),cudaMemcpyHostToDevice) != 0) {
		std::cout << "cudaMemcpy Iin failed!" << std::endl;
		exit(-1);
	}
	if (cudaMemcpy(*d_W, *h_W, F * rho * B * sizeof(__half),cudaMemcpyHostToDevice) != 0) {
		std::cout << "cudaMemcpy W failed!" << std::endl;
		exit(-1);
	}

	if (cudaMemcpy(*d_Iin_b, *h_Iin_b, F * sizeof(__half*),cudaMemcpyHostToDevice) != 0) {
		std::cout << "cudaMemcpy Iin_b failed!" << std::endl;
		exit(-1);
	}
	if (cudaMemcpy(*d_W_b, *h_W_b, F * sizeof(__half*),cudaMemcpyHostToDevice) != 0) {
		std::cout << "cudaMemcpy W_b failed!" << std::endl;
		exit(-1);
	}
	if (cudaMemcpy(*d_Iout_b, *h_Iout_b, F * sizeof(__half*),cudaMemcpyHostToDevice) != 0) {
		std::cout << "cudaMemcpy Iout_b failed!" << std::endl;
		exit(-1);
	}
}


void DtoH_copy(__half **h_Iout, __half ***h_Iout_b, __half **d_Iout, __half ***d_Iout_b, size_t F, size_t B, size_t T) {
	// TODO figure out if I need to pass in pointer to arrays or not, and if the args passed into cudaMemcpy are correct (dereference?)
	if (cudaMemcpy(*h_Iout, *d_Iout, F * T * B * sizeof(__half),cudaMemcpyDeviceToHost) != 0) {
		std::cout << "cudaMemcpy Iout failed!" << std::endl;
		exit(-1);
	}

	// TODO maybe copy the Iout_b too somehow
}


void free_memory(__half *d_Iin, __half *d_W, __half *d_Iout, __half **d_Iin_b, __half **d_W_b, __half **d_Iout_b, __half *h_Iin, __half *h_W, __half *h_Iout, __half **h_Iin_b, __half **h_W_b, __half **h_Iout_b, size_t F) {
	// TODO free d_Iout_b[f] for all f
	cudaFree(d_Iin);
	cudaFree(d_W);
	cudaFree(d_Iout);
	cudaFree(d_Iin_b);
	cudaFree(d_W_b);
	cudaFree(d_Iout_b);

	free(h_Iin_b);
	free(h_W_b);
	free(h_Iout_b);
	free(h_Iin);
	free(h_W);
	free(h_Iout);
}


int main( int argc, char *argv[] ) {
	// int num_runs = 100;
	size_t F, B, rho, T;
	
	F = 256;
	B = 5000;
	rho = 48 * 48;
	T = 52;
	if ( argc == 2 ) {
		if (strcmp(argv[1], "PF") == 0) 
		{
			B = 800;
			rho = 16 * 24;
			T = 52;
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
	if (cublasCreate(&handle) != 0) {
		std::cout << "cublasCreate failed!" << std::endl;
		return -1;
	}
	
	__half alpha = 1;
	__half beta = 0;
	__half *h_Iin, *h_W, *h_Iout;
	__half **h_Iin_b, **h_W_b, **h_Iout_b;

	__half *d_Iin, *d_W, *d_Iout;
	__half **d_Iin_b, **d_W_b, **d_Iout_b;

	allocate_memory(&h_Iin, &h_W, &h_Iout, &h_Iin_b, &h_W_b, &h_Iout_b, &d_Iin, &d_W, &d_Iout, &d_Iin_b, &d_W_b, &d_Iout_b, F, B, rho, T);

	// for testing, delete later
	if (argc == 2 and strcmp(argv[1], "test") == 0) {
		fill_manual(h_Iin, h_W, h_Iin_b, h_W_b, F, B, rho, T);
		
	} else {
		// temp fill with fake data
		fill_ones(h_Iin, T * rho * F);
		fill_ones(h_W, rho * B * F);
	}

	for (size_t f = 0; f < F; f++) { // for batched, set pointers to device memory in prep for copy
		(h_Iin_b)[f] = d_Iin + f * T * rho;

		(h_W_b)[f] = d_W + f * rho * B;

		__half* temp;
		if (cudaMalloc(&temp, T * B * sizeof(__half)) != 0) {
			std::cout << "cudaMalloc Iout_b[f] failed!" << std::endl;
			exit(-1);
		}
		(h_Iout_b)[f] = temp;
	} 

	// if (argc == 2 and (strcmp(argv[1], "test") == 0 or strcmp(argv[1], "testones") == 0)) { // can delete later
	// 	std::cout << "Iin =" << std::endl;
	// 	print_matrix(h_Iin, T, rho, F);
	// 	std::cout << "W =" << std::endl;
	// 	print_matrix(h_W, B, rho, F);
	// }

	HtoD_copy(&h_Iin, &h_W, &h_Iin_b, &h_W_b, &h_Iout_b, &d_Iin, &d_W, &d_Iin_b, &d_W_b, &d_Iout_b, F, B, rho, T);

	// start timer
	cublasStatus_t err;
	cudaError_t err2;
	if ((err2 = cudaStreamSynchronize(0)) != cudaSuccess){
		std::cout << "cudaStreamSynchronize failed: " << cudaGetErrorName(err2) << std::endl;
		return -1;
	}

	for (int i = 1; i <= 1; i += 1) {
		double time0 = gettime();

		for (int r = 0; r < i; r++) {
			for (int f = 0; f < F; f++) {	
				// Multiply A and B^T (at depth f) on GPU
				if (cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, T, B, rho, &alpha, d_Iin + f * T * rho, T, d_W + f * rho * B, B, &beta, d_Iout + f * T * B, T) != 0) {
					std::cout << "cublasHgemm failed!" << std::endl;
					return -1;
				}
			}
		}

		// end timer
		if ((err2 = cudaStreamSynchronize(0)) != cudaSuccess){
			std::cout << "cudaStreamSynchronize failed: " << cudaGetErrorName(err2) << std::endl;
			return -1;
		}
		double time1 = gettime();
		std::cout << i << " cublasHgemm operations on average finished in " << (time1 - time0) / (float) i << "s" << std::endl;
	}

	for (int i = 1; i <= 1; i += 1) {
		double time2 = gettime();

		for (int r = 0; r < i; r++) {
			if ((err = cublasHgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T, T, B, rho, &alpha, d_Iin_b, T, d_W_b, B, &beta, d_Iout_b, T, (int) F)) != 0) {
				std::cout << "cublasHgemmBatched failed: " << cublasGetStatusString(err) << std::endl;
				return -1;
			}
		}

		// end timer
		if ((err2 = cudaStreamSynchronize(0)) != cudaSuccess){
			std::cout << "cudaStreamSynchronize failed: " << cudaGetErrorName(err2) << std::endl;
			return -1;
		}
		double time3 = gettime();
		std::cout << i << " cublasHgemmBatched operations on average finished in " << (time3 - time2) / (float) i << "s" << std::endl;
	}

	// std::cout << "Temporarily ignoring seg fault after this line -- concerned with timing of operations" << std::endl;
	// DtoH_copy(&h_Iout, &h_Iout_b, &d_Iout, &d_Iout_b, F, B, T);

	// if (argc == 2 and (strcmp(argv[1], "test") == 0 or strcmp(argv[1], "testones") == 0)) { // can delete later
	// 	// Print Result 
	// 	std::cout << "Iout =" << std::endl;
	// 	print_matrix(h_Iout, T, B, F);
	// 	std::cout << "Iout_b =" << std::endl;
	// 	print_matrix_b(h_Iout_b, T, B, F);
	// }

	// Destroy the handle
	cublasDestroy(handle);

	// Free memory
	free_memory(d_Iin, d_W, d_Iout, d_Iin_b, d_W_b, d_Iout_b, h_Iin, h_W, h_Iout, h_Iin_b, h_W_b, h_Iout_b, F);
	return 0;
}
