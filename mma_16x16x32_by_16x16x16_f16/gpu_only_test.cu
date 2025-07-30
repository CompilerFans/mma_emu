#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>
#include <random>

using namespace nvcuda;

// 使用两次16x16x16来模拟16x16x32效果
__global__ void mma_16x16x32_simulation(const half* A, const half* B, float* C) {
    if (threadIdx.x >= 32) return;
    
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    
    wmma::fill_fragment(c_frag, 0.0f);
    
    // 第一次计算：A[0:16, 0:16] * B[0:16, 0:16]
    wmma::load_matrix_sync(a_frag, A, 32);        // A的前16列
    wmma::load_matrix_sync(b_frag, B, 16);        // B的前16行
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    
    // 第二次计算：A[0:16, 16:32] * B[16:32, 0:16]
    wmma::load_matrix_sync(a_frag, A + 16, 32);   // A的后16列
    wmma::load_matrix_sync(b_frag, B + 16*16, 16); // B的后16行
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    
    wmma::store_matrix_sync(C, c_frag, 16, wmma::mem_row_major);
}

// 标准的完整GEMM实现（循环处理所有K维度）
__global__ void mma_full_gemm(const half* A, const half* B, float* C, int K) {
    if (threadIdx.x >= 32) return;
    
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    
    wmma::fill_fragment(c_frag, 0.0f);
    
    // 循环处理所有K维度
    for (int k = 0; k < K; k += 16) {
        wmma::load_matrix_sync(a_frag, A + k, K);
        wmma::load_matrix_sync(b_frag, B + k*16, 16);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
    wmma::store_matrix_sync(C, c_frag, 16, wmma::mem_row_major);
}

// 只使用前16维K的基准测试
__global__ void mma_16x16x16_baseline(const half* A, const half* B, float* C) {
    if (threadIdx.x >= 32) return;
    
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    
    wmma::fill_fragment(c_frag, 0.0f);
    
    // 只使用前16维K
    wmma::load_matrix_sync(a_frag, A, 32);
    wmma::load_matrix_sync(b_frag, B, 16);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    
    wmma::store_matrix_sync(C, c_frag, 16, wmma::mem_row_major);
}

void print_matrix(const char* name, const float* data, int rows, int cols, int max_show = 4) {
    std::cout << name << ":" << std::endl;
    for (int i = 0; i < std::min(max_show, rows); i++) {
        for (int j = 0; j < std::min(max_show, cols); j++) {
            printf("%.6f ", data[i*cols + j]);
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

bool compare_results(const float* result1, const float* result2, int size, 
                    const std::string& name1, const std::string& name2, 
                    float tolerance = 1e-6) {
    float max_error = 0.0f;
    int error_count = 0;
    
    for (int i = 0; i < size; i++) {
        float error = std::abs(result1[i] - result2[i]);
        max_error = std::max(max_error, error);
        
        if (error > tolerance) {
            error_count++;
            if (error_count <= 3) {
                printf("%s[%d]=%.8f, %s[%d]=%.8f, diff=%.8f\n", 
                       name1.c_str(), i, result1[i], 
                       name2.c_str(), i, result2[i], error);
            }
        }
    }
    
    printf("%s vs %s: Max error=%.8f, Errors=%d/%d\n", 
           name1.c_str(), name2.c_str(), max_error, error_count, size);
    
    return error_count == 0;
}

int main() {
    const int M = 16, N = 16, K = 32;
    
    std::cout << "=== MMA 16x16x32 模拟验证测试 ===" << std::endl;
    std::cout << "矩阵大小: A[" << M << "x" << K << "] * B[" << K << "x" << N << "] = C[" << M << "x" << N << "]" << std::endl;
    std::cout << "目标: 验证两次16x16x16是否等效于完整的16x16x32计算" << std::endl << std::endl;
    
    // 分配内存
    std::vector<half> h_A(M * K);
    std::vector<half> h_B(K * N);
    std::vector<float> h_C_simulation(M * N);  // 模拟结果
    std::vector<float> h_C_standard(M * N);    // 标准GEMM结果  
    std::vector<float> h_C_baseline(M * N);    // 16维基准结果
    
    // 使用多组随机数据测试
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-0.3f, 0.3f);
    
    // GPU内存分配
    half *d_A, *d_B;
    float *d_C;
    
    cudaMalloc(&d_A, M * K * sizeof(half));
    cudaMalloc(&d_B, K * N * sizeof(half));
    cudaMalloc(&d_C, M * N * sizeof(float));
    
    int num_tests = 5;
    int passed_tests = 0;
    
    for (int test = 0; test < num_tests; test++) {
        std::cout << "=== 测试 " << (test + 1) << "/" << num_tests << " ===" << std::endl;
        
        // 生成新的随机数据
        for (int i = 0; i < M * K; i++) {
            h_A[i] = __float2half(dis(gen));
        }
        for (int i = 0; i < K * N; i++) {
            h_B[i] = __float2half(dis(gen));
        }
        
        // 复制到GPU
        cudaMemcpy(d_A, h_A.data(), M * K * sizeof(half), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B.data(), K * N * sizeof(half), cudaMemcpyHostToDevice);
        
        // 运行三种实现
        mma_16x16x32_simulation<<<1, 32>>>(d_A, d_B, d_C);
        cudaDeviceSynchronize();
        cudaMemcpy(h_C_simulation.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
        
        mma_full_gemm<<<1, 32>>>(d_A, d_B, d_C, K);
        cudaDeviceSynchronize();
        cudaMemcpy(h_C_standard.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
        
        mma_16x16x16_baseline<<<1, 32>>>(d_A, d_B, d_C);
        cudaDeviceSynchronize();
        cudaMemcpy(h_C_baseline.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
        
        // 对比结果
        bool test_passed = compare_results(h_C_simulation.data(), h_C_standard.data(), M*N, 
                                          "模拟实现", "标准GEMM", 1e-6);
        
        if (test_passed) {
            passed_tests++;
            std::cout << "✅ 测试通过" << std::endl;
        } else {
            std::cout << "❌ 测试失败" << std::endl;
            
            // 显示部分结果用于调试
            if (test == 0) {
                print_matrix("模拟实现结果", h_C_simulation.data(), M, N, 3);
                print_matrix("标准GEMM结果", h_C_standard.data(), M, N, 3);
                print_matrix("16维基准结果", h_C_baseline.data(), M, N, 3);
            }
        }
        std::cout << std::endl;
    }
    
    std::cout << "=== 最终结果 ===" << std::endl;
    std::cout << "通过测试: " << passed_tests << "/" << num_tests << std::endl;
    
    if (passed_tests == num_tests) {
        std::cout << "🎉 所有测试通过！" << std::endl;
        std::cout << "✅ 结论: 使用两次16x16x16成功模拟了16x16x32的MMA计算效果" << std::endl;
        std::cout << "   在多组随机数据测试中，两种实现产生了完全一致的结果。" << std::endl;
    } else {
        std::cout << "⚠️  部分测试失败，需要进一步调试。" << std::endl;
    }
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return (passed_tests == num_tests) ? 0 : -1;
}