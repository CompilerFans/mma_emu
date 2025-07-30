#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>
#include <random>

using namespace nvcuda;

// 方法1：两次16x16x16 MMA模拟
__global__ void mma_16x16x32_simulation(const half* A, const half* B, float* C) {
    if (threadIdx.x >= 32) return;
    
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    
    wmma::fill_fragment(c_frag, 0.0f);
    
    // 第一次：A[0:16, 0:16] * B[0:16, 0:16]
    wmma::load_matrix_sync(a_frag, A, 32);
    wmma::load_matrix_sync(b_frag, B, 16);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    
    // 第二次：A[0:16, 16:32] * B[16:32, 0:16]
    wmma::load_matrix_sync(a_frag, A + 16, 32);
    wmma::load_matrix_sync(b_frag, B + 16*16, 16);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    
    wmma::store_matrix_sync(C, c_frag, 16, wmma::mem_row_major);
}

// 方法2：标准MMA循环实现
__global__ void mma_standard_gemm(const half* A, const half* B, float* C, int K) {
    if (threadIdx.x >= 32) return;
    
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    
    wmma::fill_fragment(c_frag, 0.0f);
    
    for (int k = 0; k < K; k += 16) {
        wmma::load_matrix_sync(a_frag, A + k, K);
        wmma::load_matrix_sync(b_frag, B + k*16, 16);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
    wmma::store_matrix_sync(C, c_frag, 16, wmma::mem_row_major);
}

// 方法3：GPU软件实现（使用与MMA相同的数据布局）
__global__ void gpu_software_gemm(const half* A, const half* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            // 使用与MMA完全相同的数据访问模式
            // A: row_major, A[row,k] = A[row * K + k]  
            // B: col_major, B[k,col] = B[col * K + k]
            sum += __half2float(A[row * K + k]) * __half2float(B[col * K + k]);
        }
        C[row * N + col] = sum;
    }
}

void print_matrix(const char* name, const float* data, int rows, int cols) {
    std::cout << name << ":" << std::endl;
    for (int i = 0; i < std::min(4, rows); i++) {
        for (int j = 0; j < std::min(4, cols); j++) {
            printf("%.6f ", data[i*cols + j]);
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

bool compare_results(const float* result1, const float* result2, int size, 
                    const std::string& name1, const std::string& name2, 
                    float tolerance = 1e-5) {
    float max_error = 0.0f;
    int error_count = 0;
    
    for (int i = 0; i < size; i++) {
        float error = std::abs(result1[i] - result2[i]);
        max_error = std::max(max_error, error);
        if (error > tolerance) {
            error_count++;
        }
    }
    
    printf("%s vs %s: Max error=%.8f, Errors=%d/%d\n", 
           name1.c_str(), name2.c_str(), max_error, error_count, size);
    
    return error_count == 0;
}

int main() {
    const int M = 16, N = 16, K = 32;
    
    std::cout << "=== 最终正确版本：三种实现对比 ===" << std::endl;
    std::cout << "1. MMA 16x16x32 模拟（两次16x16x16）" << std::endl;
    std::cout << "2. MMA 标准循环实现" << std::endl;  
    std::cout << "3. GPU 软件实现（匹配MMA数据布局）" << std::endl << std::endl;
    
    // 分配内存
    std::vector<half> h_A(M * K);    // A: row_major
    std::vector<half> h_B(K * N);    // B: col_major
    std::vector<float> h_C_simulation(M * N);
    std::vector<float> h_C_standard(M * N);
    std::vector<float> h_C_software(M * N);
    
    // 生成随机数据
    std::random_device rd;
    std::mt19937 gen(42);  // 固定种子便于复现
    std::uniform_real_distribution<float> dis(-0.5f, 0.5f);
    
    // A矩阵：row_major
    for (int i = 0; i < M * K; i++) {
        h_A[i] = __float2half(dis(gen));
    }
    
    // B矩阵：直接按col_major布局生成
    for (int i = 0; i < K * N; i++) {
        h_B[i] = __float2half(dis(gen));
    }
    
    // GPU计算
    half *d_A, *d_B;
    float *d_C;
    
    cudaMalloc(&d_A, M * K * sizeof(half));
    cudaMalloc(&d_B, K * N * sizeof(half));
    cudaMalloc(&d_C, M * N * sizeof(float));
    
    cudaMemcpy(d_A, h_A.data(), M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), K * N * sizeof(half), cudaMemcpyHostToDevice);
    
    std::cout << "运行计算..." << std::endl;
    
    // 方法1：MMA模拟
    mma_16x16x32_simulation<<<1, 32>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C_simulation.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // 方法2：MMA标准
    mma_standard_gemm<<<1, 32>>>(d_A, d_B, d_C, K);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C_standard.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // 方法3：软件实现
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    gpu_software_gemm<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C_software.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // 显示结果
    std::cout << "\n=== 计算结果 ===" << std::endl;
    print_matrix("MMA模拟结果", h_C_simulation.data(), M, N);
    print_matrix("MMA标准结果", h_C_standard.data(), M, N);
    print_matrix("GPU软件结果", h_C_software.data(), M, N);
    
    // 精度验证
    std::cout << "=== 精度验证 ===" << std::endl;
    bool test1 = compare_results(h_C_simulation.data(), h_C_standard.data(), M*N, 
                                "MMA模拟", "MMA标准");
    bool test2 = compare_results(h_C_simulation.data(), h_C_software.data(), M*N, 
                                "MMA模拟", "GPU软件");
    bool test3 = compare_results(h_C_standard.data(), h_C_software.data(), M*N, 
                                "MMA标准", "GPU软件");
    
    std::cout << "\n=== 最终结论 ===" << std::endl;
    if (test1) {
        std::cout << "✅ MMA模拟与MMA标准完全一致！" << std::endl;
        std::cout << "🎉 两次16x16x16成功模拟16x16x32效果" << std::endl;
    }
    
    if (test2 && test3) {
        std::cout << "✅ 所有三种实现结果完全一致！" << std::endl;
        std::cout << "🔬 验证了算法的数学正确性" << std::endl;
    }
    
    bool all_passed = test1 && test2 && test3;
    std::cout << "\n总体测试: " << (all_passed ? "完全通过 🎉" : "需要调试 ⚠️") << std::endl;
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return all_passed ? 0 : -1;
}