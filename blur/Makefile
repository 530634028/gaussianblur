
# ----------------------------------------------------------------------------

# mode: cpu, gpu
mode = cpu

# ----------------------------------------------------------------------------

BUILD_DIR = _build
TESTS_DIR = tests

HEADERS = gaussian.hh mat.h utils.hh
SOURCES = gaussian.cc
SOURCES_CUDA = gaussian.cu

# ----------------------------------------------------------------------------

# include: cuda
CUDA_INCLUD = -I/usr/local/cuda/include
CUDA_LIB = -L/usr/local/cuda/lib64
CUDA_ARCH = -gencode arch=compute_50,code=sm_50

# g++
CC = g++
CC_FLAGS = -std=c++11
# cuda
NVCC = nvcc
NVCC_FLAGS = $(CUDA_INCLUDE) $(CUDA_ARCH)
NVCC_LDFLAGS = $(CUDA_LIB) -lcublas

# ----------------------------------------------------------------------------

all: 
	cat log

cpu:
	$(CC) -o $(BUILD_DIR)/test_gaussian_cpu $(TESTS_DIR)/test_gaussian_cpu.cc $(SOURCES) $(HEADERS) $(CC_FLAGS)
	$(BUILD_DIR)/test_gaussian_cpu > log
	cat log

cuda:
	$(NVCC) -o $(BUILD_DIR)/test_gaussian_gpu $(TESTS_DIR)/test_gaussian_gpu.cu $(SOURCES_CUDA) $(CUDA_INCLUDE) $(CUDA_LIB)
	$(BUILD_DIR)/test_gaussian_gpu > log
	cat log

clean:
	rm -rf $(BUILD_DIR)/*
