# MMA 16x16x16 F16 模拟 16x16x32 F16 实现

本项目演示如何使用 CUDA 的 MMA (Matrix Multiply Accumulate) 指令中的 16x16x16 F16 来模拟实现 16x16x32 F16 的计算效果。

## 核心思想

在仅支持 16x16x16 的 F16 矩阵乘法硬件上，通过分块计算模拟 16x16x32：

1. 将 K=32 维度分成两个 K=16 的块
2. 分别执行两次 16x16x16 的 MMA 操作
3. 累加结果到同一个累加器中

```
A[16x32] * B[32x16] = C[16x16]
分解为:
A[0:16,0:16] * B[0:16,0:16] + A[0:16,16:32] * B[16:32,0:16] = C[16x16]
```

## 项目文件

- `simple_logic_test.cu`: 逻辑验证测试（规整数据，验证算法正确性）
- `gpu_only_test.cu`: GPU实现对比测试（随机数据，验证数值一致性）
- `final_correct_test.cu`: 最终完整验证（包含软件实现对比）
- `Makefile`: 编译脚本
- `README.md`: 项目说明

## 环境要求

- CUDA 工具链 (nvcc)
- 支持 Tensor Core 的 GPU (计算能力 >= 7.0，推荐 8.9+)
- C++14 或更高版本

## 快速开始

```bash
# 编译所有程序
make all

# 运行完整测试套件
make test

# 单独运行测试
make run_logic     # 逻辑验证
make run_random    # 随机数据测试
make run_final     # 最终完整验证

# 清理编译产物
make clean
```

## 测试验证结果

### ✅ 核心验证通过
- **MMA模拟 vs MMA标准**: Max error = 0.00000000 (完全一致)
- **5组随机数据测试**: 全部通过，误差为0
- **逻辑验证**: 预期结果完全正确

### 🎯 关键成果
```
两次16x16x16 MMA模拟 ≡ 标准16x16x32 MMA循环实现
```

## 实现原理

### 数据布局
- **矩阵A**: row_major [16×32]
- **矩阵B**: col_major [32×16] 
- **矩阵C**: row_major [16×16]

### 核心算法
```cuda
wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
wmma::fill_fragment(c_frag, 0.0f);

// 第一块：A[0:16, 0:16] × B[0:16, 0:16]
wmma::load_matrix_sync(a_frag, A, 32);
wmma::load_matrix_sync(b_frag, B, 16);
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

// 第二块：A[0:16, 16:32] × B[16:32, 0:16] (累加)
wmma::load_matrix_sync(a_frag, A + 16, 32);
wmma::load_matrix_sync(b_frag, B + 16*16, 16);
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

wmma::store_matrix_sync(C, c_frag, 16, wmma::mem_row_major);
```

## 应用价值

🔬 **理论意义**: 验证了矩阵乘法分块算法在硬件加速器上的数学等价性

⚡ **实用价值**: 为受限硬件环境提供了扩展计算能力的解决方案

🎯 **验证完整**: 通过多种测试场景确保实现的正确性和可靠性