#include "mask.cuh"
#define DBL_EPSILON 2.2204460492503131e-16


void generate_random_ones(uint32_t *arr, size_t size) {
    srand(time(NULL)); 
    for (int i = 0; i < size; i++) {
        arr[i] = rand() & 0x1;
    }
}


// helper function for generate_noise_array
int generate_gaussian_number(int mean, int stdDev) {
    int z0, z1;

    int u1, u2;
    do {
        u1 = rand() / (int)RAND_MAX;
        u2 = rand() / (int)RAND_MAX;
    } while (u1 <= DBL_EPSILON);

    z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
    z1 = sqrt(-2.0 * log(u1)) * sin(2.0 * M_PI * u2);

    return (int)round(z0 * stdDev + mean);
}


// helper function for generate_noise_array
int generate_lorentzian_number(int location, int scale) {
    int u = rand() / (int)RAND_MAX;
    return (int)round(location + scale * tan(M_PI * (u - 0.5)));
}


void generate_noise_array(uint32_t* arr, size_t size) {
    srand(time(NULL)); // Seed the random number generator with the current time

    int mean = 0.0;  // Mean of the Gaussian distribution
    int stdDev = 1.0; // Standard deviation of the Gaussian distribution

    int location = 0.0; // Location parameter of the Lorentzian distribution
    int scale = 1.0; // Scale parameter of the Lorentzian distribution

    for (int i = 0; i < size / 2; i++) {
        uint32_t e = generate_gaussian_number(mean, stdDev) << 28;
        e = e ^ ((generate_gaussian_number(mean, stdDev) & 0xf) << 24);
        e = e ^ ((generate_gaussian_number(mean, stdDev) & 0xf) << 20);
        e = e ^ ((generate_gaussian_number(mean, stdDev) & 0xf) << 16);
        e = e ^ ((generate_gaussian_number(mean, stdDev) & 0xf) << 12);
        e = e ^ ((generate_gaussian_number(mean, stdDev) & 0xf) << 8);
        e = e ^ ((generate_gaussian_number(mean, stdDev) & 0xf) << 4);
        e = e ^ (generate_gaussian_number(mean, stdDev) & 0xf);

        arr[i] = e;
    }

    for (int i = size / 2; i < size; i++) {
        uint32_t e = generate_lorentzian_number(location, scale) << 28;
        e = e ^ ((generate_lorentzian_number(location, scale) & 0xf) << 24);
        e = e ^ ((generate_lorentzian_number(location, scale) & 0xf) << 20);
        e = e ^ ((generate_lorentzian_number(location, scale) & 0xf) << 16);
        e = e ^ ((generate_lorentzian_number(location, scale) & 0xf) << 12);
        e = e ^ ((generate_lorentzian_number(location, scale) & 0xf) << 8);
        e = e ^ ((generate_lorentzian_number(location, scale) & 0xf) << 4);
        e = e ^ (generate_lorentzian_number(location, scale) & 0xf);

        arr[i] = e;
    }
}


__device__ float d_M_func(float mu) {
    return 0;
}

__device__ float d_V_func(float mu) {
    return 1;
}


__global__ void __launch_bounds__(1024, 1) mask(uint32_t *R, uint32_t *W, uint *S1, uint *S2, size_t N, size_t D, size_t T_bar, size_t F, float mu_min, float N_good_min, float sigma, float *SK, float *mean_SK, float *var_SK) {
    int th_col = threadIdx.x % 32; // thread num
    int th_row = threadIdx.x / 32; // warp num: 1 t each

    int N_good = 0;
    // sum up W's corresponding to feeds in this thread
    for (int i = th_col; i < D * 2; i = i + 32) {
        N_good += W[i];
    }

    // sum up N_good from all threads
    for (int p = 0; p <= 4; p++) { N_good = N_good + __shfl_sync(0xffffffff, N_good, threadIdx.x ^ (1 << p)); }

    // check if there are enough good feeds to continue
    if (N_good < N_good_min) { 
        R[blockIdx.x * T_bar + blockIdx.y * 32 + th_row] = 0; 
        printf("Not enough good feeds\n");
        return; 
    }

    // declare what we need
    float mu;
    float sum, mean_sum, var_sum;
    float S2_tilde;
    int s_idx;

    sum = 0;
    mean_sum = 0;
    var_sum = 0;

    // loop over feeds in this thread, calc sum, mean_sum and var_sum
    for (int i = th_col; i < D * 2; i = i + 32) {
        s_idx = blockIdx.y * 32 * F * D * 2 + blockIdx.x * D * 2 + th_row * F * D * 2 + i;
        mu = float(S1[s_idx]) / (float) N;

        if (mu >= mu_min) { 
            S2_tilde = float(S2[s_idx]) / (mu * mu); 
            sum += (float) W[i] * (S2_tilde / (float) N - 1);
            mean_sum += (float) W[i] * d_M_func(mu);
            var_sum += (float) W[i] * d_V_func(mu);
        }
    }

    int source;
    for (int p = 0; p <= 4; p++) {
        source = threadIdx.x ^ (1 << p);
        sum = sum + __shfl_sync(0xffffffff, sum, source);
        mean_sum = mean_sum + __shfl_sync(0xffffffff, mean_sum, source);
        var_sum = var_sum + __shfl_sync(0xffffffff, var_sum, source);
    }

    if (th_col == 0) {
        // some constants
        float frac = (1 / N_good) * ((float) N + 1) / ((float) N - 1);
        float mean_frac = 1 + 1 / N_good;
        float var_frac = 4 / (N_good * N_good * (float) N);

        // final numbers we need
        float sk = frac * sum;
        float mean_sk = mean_frac * mean_sum;
        float var_sk = var_frac * var_sum;

        // fill R with true or false
        R[blockIdx.x * T_bar + blockIdx.y * 32 + th_row] = (abs(sk - mean_sk) <= sigma * sqrt(var_sk) ? true: false);

        // TODO delete later
        // SK[blockIdx.x * T_bar + blockIdx.y * 32 + th_row] = sk;
        // mean_SK[blockIdx.x * T_bar + blockIdx.y * 32 + th_row] = mean_sk;
        // var_SK[blockIdx.x * T_bar + blockIdx.y * 32 + th_row] = var_sk;
        // R[blockIdx.x * T_bar + blockIdx.y * 32 + th_row] = N_good;
    }
}
