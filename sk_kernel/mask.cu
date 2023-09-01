#include "mask.cuh"

void generate_random_ones(bool *arr, size_t size) {
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

__device__ float d_M_func(float mu) {
    return 0;
}

__device__ float d_V_func(float mu) {
    return 1;
}


__global__ void mask(bool *R, bool *W, float4 *S1, float4 *S2, size_t N, size_t D, size_t T_bar, size_t F, float mu_min, float N_good_min) {
    int th_col = threadIdx.x % 32;
    int th_row = threadIdx.x / 32;

    int N_good = 0;
    // sum up W's corresponding to feeds in this thread
    for (int i = th_row; i < (th_row) + D * 2 / 32; i++) {
        N_good += W[i];
    }

    // sum up N_good from all threads
    for (int p = 0; p <= 4; p++) {
        N_good = N_good + __shfl_sync(-1, N_good, pow(th_col, p));
    }

    // check if there are enough good feeds to continue
    if (N_good < N_good_min) { 
        R[blockIdx.x * T_bar / 32 + blockIdx.y * 32 + th_row] = 0; 
        return; 
    }

    // declare what we need
    float mu_0, mu_1, mu_2, mu_3;
    float sum, mean_sum, var_sum;
    float S2_tilde_0, S2_tilde_1, S2_tilde_2, S2_tilde_3;

    // loop over feeds in this thread, calc sum, mean_sum and var_sum
    for (int i = threadIdx.x / (32 * 4); i < (threadIdx.x / (32 * 4)) + D * 2 / (32 * 4); i++) {
        mu_0 = S1[i].x / (float) N;
        mu_1 = S1[i].y / (float) N;
        mu_2 = S1[i].z / (float) N;
        mu_3 = S1[i].w / (float) N;

        sum = 0;
        mean_sum = 0;
        var_sum = 0;

        if (mu_0 >= mu_min) { 
            S2_tilde_0 = S2[i].x / (mu_0 * mu_0); 
            sum += W[i * 4] * S2_tilde_0;
            mean_sum += W[i * 4] * d_M_func(mu_0);
            var_sum += W[i * 4] * d_V_func(mu_0);
        }
        if (mu_1 >= mu_min) { 
            S2_tilde_1 = S2[i].y / (mu_1 * mu_1); 
            sum += W[i * 4 + 1] * S2_tilde_1;
            mean_sum += W[i * 4 + 1] * d_M_func(mu_1);
            var_sum += W[i * 4 + 1] * d_V_func(mu_1);
        }
        if (mu_2 >= mu_min) { 
            S2_tilde_2 = S2[i].z / (mu_2 * mu_2); 
            sum += W[i * 4 + 2] * S2_tilde_2;
            mean_sum += W[i * 4 + 2] * d_M_func(mu_2);
            var_sum += W[i * 4 + 2] * d_V_func(mu_2);
        }
        if (mu_3 >= mu_min) { 
            S2_tilde_3 = S2[i].w / (mu_3 * mu_3);
            sum += W[i * 4 + 3] * S2_tilde_3;
            mean_sum += W[i * 4 + 3] * d_M_func(mu_3);
            var_sum += W[i * 4 + 3] * d_V_func(mu_3);
        }

        for (int p = 0; p <= 4; p++) {
            sum = sum + __shfl_sync(-1, sum, pow(th_col, p));
            mean_sum = mean_sum + __shfl_sync(-1, mean_sum, pow(th_col, p));
            var_sum = var_sum + __shfl_sync(-1, var_sum, pow(th_col, p));
        }
    }

    if (th_col == 0) {
        // some constants
        float frac = 1 / N_good + (N + 1) / (N - 1);
        float mean_frac = 1 + 1 / N_good;
        float var_frac = 4 / (N_good * N_good * N);

        // final numbers we need
        float sk = frac * sum;
        float mean_sk = mean_frac * mean_sum;
        float var_sk = var_frac * var_sum;

        // fill R with true or false
        R[blockIdx.x * T_bar + blockIdx.y * 32 + th_row] = abs(sk - mean_sk) <= 5 * sqrt(var_sk) ? true: false;
    }
}