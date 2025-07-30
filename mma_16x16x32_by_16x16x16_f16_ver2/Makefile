NVCC = nvcc
NVCCFLAGS = -arch=sm_89 -std=c++14 -O3
INCLUDES = 
LIBS = 

# 目标文件
TARGETS = mma_correct_impl

# 默认目标
all: $(TARGETS)

mma_correct_impl: mma_correct_impl.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $< $(LIBS)


# 运行逻辑测试
run: mma_correct_impl
	./mma_correct_impl

# 运行所有测试
test: $(TARGETS)
	@echo "=== 运行逻辑验证测试 ==="
	./mma_correct_impl

# 清理
clean:
	rm -f $(TARGETS)

# 检查GPU信息
gpu_info:
	nvidia-smi
	nvcc --version

.PHONY: all run test clean gpu_info
