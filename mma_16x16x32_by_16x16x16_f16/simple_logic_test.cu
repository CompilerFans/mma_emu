#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>

using namespace nvcuda;

// 使用两次16x16x16来模拟16x16x32
__global__ void mma_16x16x32_simulation(const half* A, const half* B, float* C) {
    if (threadIdx.x >= 32) return;
    
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    
    wmma::fill_fragment(c_frag, 0.0f);
    
    // 第一次计算：A[16x16] * B[16x16] (使用K的前16维)
    wmma::load_matrix_sync(a_frag, A, 32);        // A的前16列，stride=32
    wmma::load_matrix_sync(b_frag, B, 16);        // B的前16行，stride=16
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    
    // 第二次计算：A[16x16] * B[16x16] (使用K的后16维)
    wmma::load_matrix_sync(a_frag, A + 16, 32);   // A的后16列，offset+16
    wmma::load_matrix_sync(b_frag, B + 16*16, 16); // B的后16行，offset+16*16
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    
    wmma::store_matrix_sync(C, c_frag, 16, wmma::mem_row_major);
}

// 基准：只使用前16维的K
__global__ void mma_16x16x16_baseline(const half* A, const half* B, float* C) {
    if (threadIdx.x >= 32) return;
    
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    
    wmma::fill_fragment(c_frag, 0.0f);
    
    wmma::load_matrix_sync(a_frag, A, 32);
    wmma::load_matrix_sync(b_frag, B, 16);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    
    wmma::store_matrix_sync(C, c_frag, 16, wmma::mem_row_major);
}

void print_matrix(const char* name, const float* data, int rows, int cols, int max_show = 4) {
    std::cout << name << " (" << rows << "x" << cols << "):" << std::endl;
    for (int i = 0; i < std::min(max_show, rows); i++) {
        for (int j = 0; j < std::min(max_show, cols); j++) {
            printf("%.3f ", data[i*cols + j]);
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void print_matrix_half(const char* name, const half* data, int rows, int cols, int max_show = 4) {
    std::cout << name << " (" << rows << "x" << cols << "):" << std::endl;
    for (int i = 0; i < std::min(max_show, rows); i++) {  
        for (int j = 0; j < std::min(max_show, cols); j++) {
            printf("%.1f ", __half2float(data[i*cols + j]));
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void cpu_gemm(const half* A, const half* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += __half2float(A[i*32 + k]) * __half2float(B[k*16 + j]);
            }
            C[i*N + j] = sum;
        }
    }
}

int main() {
    const int M = 16, N = 16, K1 = 16, K2 = 32;
    
    std::cout << "=== 测试逻辑：使用两次16x16x16模拟16x16x32 ===" << std::endl;
    
    // 创建规整的测试数据
    std::vector<half> h_A(M * K2);  // 16x32
    std::vector<half> h_B(K2 * N);  // 32x16
    std::vector<float> h_C_16(M * N);   // 结果：16维K
    std::vector<float> h_C_32(M * N);   // 结果：32维K
    std::vector<float> h_C_cpu_16(M * N);  // CPU参考：16维
    std::vector<float> h_C_cpu_32(M * N);  // CPU参考：32维
    
    // 初始化矩阵A：每行递增
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K2; j++) {
            h_A[i*K2 + j] = __float2half(j < K1 ? 1.0f : 2.0f);  // 前16维=1，后16维=2
        }
    }
    
    // 初始化矩阵B：每列递增
    for (int i = 0; i < K2; i++) {
        for (int j = 0; j < N; j++) {
            h_B[i*N + j] = __float2half(i < K1 ? 1.0f : 3.0f);  // 前16行=1，后16行=3
        }
    }
    
    std::cout << "数据规律:" << std::endl;
    std::cout << "A: 前16列=1.0, 后16列=2.0" << std::endl;
    std::cout << "B: 前16行=1.0, 后16行=3.0" << std::endl;
    std::cout << "预期结果:" << std::endl;
    std::cout << "16维K: 1*1*16 = 16.0" << std::endl;
    std::cout << "32维K: 1*1*16 + 2*3*16 = 16 + 96 = 112.0" << std::endl;
    std::cout << std::endl;
    
    print_matrix_half("A", h_A.data(), M, K2, 4);
    print_matrix_half("B", h_B.data(), K2, N, 4);
    
    // GPU计算
    half *d_A, *d_B;
    float *d_C;
    
    cudaMalloc(&d_A, M * K2 * sizeof(half));
    cudaMalloc(&d_B, K2 * N * sizeof(half));
    cudaMalloc(&d_C, M * N * sizeof(float));
    
    cudaMemcpy(d_A, h_A.data(), M * K2 * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), K2 * N * sizeof(half), cudaMemcpyHostToDevice);
    
    // 测试16x16x16（只使用前16维）
    mma_16x16x16_baseline<<<1, 32>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C_16.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // 测试16x16x32模拟（使用全部32维）
    mma_16x16x32_simulation<<<1, 32>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C_32.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // CPU参考计算
    cpu_gemm(h_A.data(), h_B.data(), h_C_cpu_16.data(), M, N, K1);  // 只用前16维
    cpu_gemm(h_A.data(), h_B.data(), h_C_cpu_32.data(), M, N, K2);  // 用全部32维
    
    // 显示结果
    std::cout << "=== 结果对比 ===" << std::endl;
    print_matrix("GPU 16维结果", h_C_16.data(), M, N, 4);
    print_matrix("CPU 16维参考", h_C_cpu_16.data(), M, N, 4);
    
    print_matrix("GPU 32维结果", h_C_32.data(), M, N, 4);
    print_matrix("CPU 32维参考", h_C_cpu_32.data(), M, N, 4);
    
    // 检查正确性
    bool test16_ok = (std::abs(h_C_16[0] - 16.0f) < 1e-3);
    bool test32_ok = (std::abs(h_C_32[0] - 112.0f) < 1e-3);
    bool cpu16_ok = (std::abs(h_C_cpu_16[0] - 16.0f) < 1e-3);
    bool cpu32_ok = (std::abs(h_C_cpu_32[0] - 112.0f) < 1e-3);
    
    std::cout << "=== 验证结果 ===" << std::endl;
    printf("GPU 16维: %.3f (预期16.0) %s\n", h_C_16[0], test16_ok ? "✓" : "✗");
    printf("GPU 32维: %.3f (预期112.0) %s\n", h_C_32[0], test32_ok ? "✓" : "✗");
    printf("CPU 16维: %.3f (预期16.0) %s\n", h_C_cpu_16[0], cpu16_ok ? "✓" : "✗");
    printf("CPU 32维: %.3f (预期112.0) %s\n", h_C_cpu_32[0], cpu32_ok ? "✓" : "✗");
    
    bool all_passed = test16_ok && test32_ok && cpu16_ok && cpu32_ok;
    std::cout << "\n测试结果: " << (all_passed ? "全部通过 ✓" : "有错误 ✗") << std::endl;
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return all_passed ? 0 : -1;
}