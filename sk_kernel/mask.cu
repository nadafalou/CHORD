#include "mask.cuh"
#define DBL_EPSILON 2.2204460492503131e-16


void generate_random_ones(uint32_t *arr, size_t size) {
    srand(time(NULL)); 
    for (int i = 0; i < size; i++) {
        arr[i] = rand() & 0x1;
    }
}


void generate_random_float4(float4 *arr, size_t size) {
    srand(time(NULL)); 
    for (int i = 0; i < size; i++) {
        float4 a;
        a.x = rand() % (65 + 1 - 0) + 0; //(max_number + 1 - minimum_number) + minimum_number
        a.y = rand() % (65 + 1 - 0) + 0;
        a.z = rand() % (65 + 1 - 0) + 0;
        a.w = rand() % (65 + 1 - 0) + 0;

        arr[i] = a;
        // printf("%lu \n", (unsigned long) e);
    }
}


int generate_gaussian_number(double mean, double stdDev) {
    static double z0, z1;
    // static int generate;
    // generate = !generate;

    // if (!generate)
    //     return z1 * stdDev + mean;

    double u1, u2;
    do {
        u1 = rand() / (double)RAND_MAX;
        u2 = rand() / (double)RAND_MAX;
    } while (u1 <= DBL_EPSILON);

    z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
    z1 = sqrt(-2.0 * log(u1)) * sin(2.0 * M_PI * u2);

    return (int)round(z0 * stdDev + mean);
}


int generate_lorentzian_number(double location, double scale) {
    double u = rand() / (double)RAND_MAX;
    return (int)round(location + scale * tan(M_PI * (u - 0.5)));
}


void generate_noise_array(uint32_t* arr, size_t size) {
    srand(time(NULL)); // Seed the random number generator with the current time

    double mean = 0.0;  // Mean of the Gaussian distribution
    double stdDev = 1.0; // Standard deviation of the Gaussian distribution

    double location = 0.0; // Location parameter of the Lorentzian distribution
    double scale = 1.0; // Scale parameter of the Lorentzian distribution

    for (int i = 0; i < size; i++) {
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

    // for (int i = size / 2; i < size; i++) {
    //     uint32_t e = generate_lorentzian_number(location, scale) << 28;
    //     e = e ^ ((generate_lorentzian_number(location, scale) & 0xf) << 24);
    //     e = e ^ ((generate_lorentzian_number(location, scale) & 0xf) << 20);
    //     e = e ^ ((generate_lorentzian_number(location, scale) & 0xf) << 16);
    //     e = e ^ ((generate_lorentzian_number(location, scale) & 0xf) << 12);
    //     e = e ^ ((generate_lorentzian_number(location, scale) & 0xf) << 8);
    //     e = e ^ ((generate_lorentzian_number(location, scale) & 0xf) << 4);
    //     e = e ^ (generate_lorentzian_number(location, scale) & 0xf);

    //     arr[i] = e;
    // }
}


__device__ float d_M_func(float mu) {
    return 0;
}

__device__ float d_V_func(float mu) {
    return 1;
}


__global__ void __launch_bounds__(1024, 1) mask(uint32_t *R, uint32_t *W, float *S1, float *S2, size_t N, size_t D, size_t T_bar, size_t F, float mu_min, float N_good_min, float sigma, float *SK, float *mean_SK, float *var_SK) {
    int th_col = threadIdx.x % 32; // thread num
    int th_row = threadIdx.x / 32; // warp num: 1 t each

    float N_good = 0;
    // sum up W's corresponding to feeds in this thread
    for (int i = th_col; i < D * 2; i = i + 32) {
        N_good += W[i];
    }

    // sum up N_good from all threads
    for (int p = 0; p <= 4; p++) { N_good = N_good + __shfl_sync(0xffffffff, N_good, threadIdx.x ^ (int) pow(2, p)); }

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

    sum = 0;
    mean_sum = 0;
    var_sum = 0;

    // loop over feeds in this thread, calc sum, mean_sum and var_sum
    for (int i = th_col; i < D * 2; i = i + 32) {
        mu = S1[i] / (float) N;

        if (mu >= mu_min) { 
            S2_tilde = S2[i] / (mu * mu); 
            sum += (float) W[i] * (S2_tilde / (float) N - 1);
            mean_sum += (float) W[i] * d_M_func(mu);
            var_sum += (float) W[i] * d_V_func(mu);
        }
    }

    for (int p = 0; p <= 4; p++) {
        sum = sum + __shfl_sync(0xffffffff, sum, threadIdx.x ^ (int) pow(2, p));
        mean_sum = mean_sum + __shfl_sync(0xffffffff, mean_sum, threadIdx.x ^ (int) pow(2, p));
        var_sum = var_sum + __shfl_sync(0xffffffff, var_sum, threadIdx.x ^ (int) pow(2, p));
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

        // temp
        SK[blockIdx.x * T_bar + blockIdx.y * 32 + th_row] = sk;
        mean_SK[blockIdx.x * T_bar + blockIdx.y * 32 + th_row] = mean_sk;
        var_SK[blockIdx.x * T_bar + blockIdx.y * 32 + th_row] = var_sk;
    }
}