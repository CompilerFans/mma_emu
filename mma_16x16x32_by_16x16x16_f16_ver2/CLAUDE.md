# 项目说明

## 项目用途
本项目演示如何在仅支持16x16x16 MMA指令的CUDA硬件上，实现16x16x32矩阵乘法的计算效果。

## 核心实现
- 文件：`mma_correct_impl.cu`
- 技术：将16x16x32分解为两个16x16x16 MMA操作并累加结果

## 数学分解
```
C = A × B
  = A[:, 0:16] × B[0:16, :] + A[:, 16:32] × B[16:32, :]
  = (16×16) × (16×16) + (16×16) × (16×16)
```

## 编译运行
```bash
nvcc -arch=sm_80 -o mma_correct_impl mma_correct_impl.cu -lcudart
./mma_correct_impl
```

## 验证结果
- ✅ 计算结果完全匹配
- ✅ 最大误差：0.000000