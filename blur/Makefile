
# ----------------------------------------------------------------------------

# mode: cpu, gpu
mode = cpu

# ----------------------------------------------------------------------------

BUILD_DIR = _build
TESTS_DIR = tests
KERNEL_DIR = kernel

# ----------------------------------------------------------------------------

# include: g++
CC_INCLUDE = -I./ -I$(TESTS_DIR) -I$(KERNEL_DIR)
# include: cuda
CUDA_INCLUD = -I/usr/local/cuda/include
CUDA_LIB = -L/usr/local/cuda/lib64 -lcuda
CUDA_ARCH = -gencode arch=compute_50,code=sm_50

# g++
CC = g++
CC_FLAGS = -std=c++11 $(CC_INCLUDE)
# cuda
NVCC = nvcc
NVCC_FLAGS = $(CC_INCLUDE) $(CUDA_INCLUDE) $(CUDA_ARCH)
NVCC_LDFLAGS = $(CUDA_LIB)

# ----------------------------------------------------------------------------


cpu:
	$(CC) -o $(BUILD_DIR)/test_gaussian_cpu $(TESTS_DIR)/test_gaussian_cpu.cc gaussian.cc $(CC_FLAGS)
	$(BUILD_DIR)/test_gaussian_cpu > log
	cat log

gpu: cuda
	$(NVCC) -o $(BUILD_DIR)/test_gaussian_gpu $(TESTS_DIR)/test_gaussian_gpu.cc gaussian.cc $(CC_FLAGS) -L$(BUILD_DIR) -lkernel $(CUDA_INCLUDE) $(CUDA_LIB)
	$(BUILD_DIR)/test_gaussian_gpu > log
	cat log

cuda:
	$(NVCC) -c $(KERNEL_DIR)/gaussian.cu -o $(BUILD_DIR)/gaussian_cu.o $(CUDA_INCLUDE) $(CUDA_LIB)
	$(NVCC) -c $(KERNEL_DIR)/kernel.cu -o $(BUILD_DIR)/kernel.o $(CUDA_INCLUDE) $(CUDA_LIB)
	$(AR) cr $(BUILD_DIR)/libkernel.a $(BUILD_DIR)/kernel.o $(BUILD_DIR)/gaussian_cu.o
	$(AR) -t $(BUILD_DIR)/libkernel.a

clean:
	rm -rf $(BUILD_DIR)/*

dummy:
	ls $(KERNEL_DIR)
