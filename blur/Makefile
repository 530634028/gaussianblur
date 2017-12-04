
# ----------------------------------------------------------------------------

# mode: cpu, gpu
mode = cpu

# ----------------------------------------------------------------------------

BUILD_DIR = _build
TESTS_DIR = tests

# ----------------------------------------------------------------------------

CC_INCLUDES = -I./ -I$(TESTS_DIR)

# ----------------------------------------------------------------------------

CUDA_INCLUDE_DIR = /usr/local/cuda/include
CUDA_LIB_DIR = /usr/local/cuda/lib64
CUDA_FLAGS = -I$(CUDA_INCLUDE_DIR)
CUDA_LIBS = -L$(CUDA_LIB_DIR) -lcuda -lcudart

# ----------------------------------------------------------------------------

CC = g++
CC_OPTS = -std=c++11 -O3 -Wall -Wextra -m64
CC_FLAGS = $(CC_OPTS) $(CC_INCLUDES)
CC_LDFLAGS = -fopenmp -lpthread 

NVCC = nvcc
NVCC_OPTS = -std=c++11 -O3 -arch=sm_50 -Xcompiler -Wall -Xcompiler -Wextra -m64
NVCC_FLAGS = $(NVCC_OPTS) -I$(CUDA_FLAGS)
NVCC_LDFLAGS = $(CUDA_LIBS)

# ----------------------------------------------------------------------------

help:
	@echo "Command List:"
	@echo "- all"
	@echo "- cpu"
	@echo "- gpu"
	@echo "- kernel"

all: cpu gpu kernel border

cpu:
	$(CC) -o $(BUILD_DIR)/test_gaussian_cpu $(TESTS_DIR)/test_gaussian_cpu.cc gaussian.cc $(CC_FLAGS) $(CC_LDFLAGS)
	$(BUILD_DIR)/test_gaussian_cpu > log
	cat log

gpu: cuda
	$(CC) -o $(BUILD_DIR)/test_gaussian_gpu $(TESTS_DIR)/test_gaussian_gpu.cc gaussian.cc $(BUILD_DIR)/gaussian_cu.o $(CC_FLAGS) $(CC_LDFLAGS) $(CUDA_FLAGS) $(CUDA_LIBS)
	$(BUILD_DIR)/test_gaussian_gpu > log
	cat log

cuda:
	$(NVCC) -c gaussian.cu -o $(BUILD_DIR)/gaussian_cu.o $(CC_INCLUDES) $(NVCC_FLAGS) $(NVCC_LDFLAGS)


kernel:
	$(CC) -o $(BUILD_DIR)/test_gaussian_kernel $(TESTS_DIR)/test_gaussian_kernel.cc gaussian.cc $(CC_FLAGS)
	$(BUILD_DIR)/test_gaussian_kernel

border:
	$(CC) -o $(BUILD_DIR)/test_border $(TESTS_DIR)/test_border.cc $(CC_FLAGS)
	$(BUILD_DIR)/test_border

clean:
	rm -rf $(BUILD_DIR)/*

dummy:
	ls $(KERNEL_DIR)
