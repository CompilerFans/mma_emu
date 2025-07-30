NVCC = nvcc
NVCCFLAGS = -arch=sm_89 -std=c++14 -O3
INCLUDES = 
LIBS = 

# 目标文件
TARGETS = simple_logic_test gpu_only_test final_correct_test

# 默认目标
all: $(TARGETS)

# 逻辑验证测试（规整数据）
simple_logic_test: simple_logic_test.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $< $(LIBS)

# GPU实现对比测试（随机数据）
gpu_only_test: gpu_only_test.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $< $(LIBS)

# 最终正确版本测试（3种实现方式）
final_correct_test: final_correct_test.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $< $(LIBS)

# 运行逻辑测试
run_logic: simple_logic_test
	./simple_logic_test

# 运行随机测试
run_random: gpu_only_test
	./gpu_only_test

# 运行最终正确测试
run_final: final_correct_test
	./final_correct_test

# 运行所有测试
test: $(TARGETS)
	@echo "=== 运行逻辑验证测试 ==="
	./simple_logic_test
	@echo ""
	@echo "=== 运行随机数据测试 ==="
	./gpu_only_test
	@echo ""
	@echo "=== 运行最终正确测试 ==="
	./final_correct_test

# 清理
clean:
	rm -f $(TARGETS)

# 检查GPU信息
gpu_info:
	nvidia-smi
	nvcc --version

.PHONY: all run_logic run_random run_final test clean gpu_info