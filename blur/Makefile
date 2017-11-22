
# ----------------------------------------------------------------------------

# mode: cpu, gpu
mode = cpu

# ----------------------------------------------------------------------------

BUILD_DIR = _build

HEADERS = gaussian.hh mat.h utils.hh
SOURCES = gaussian.cc

# ----------------------------------------------------------------------------

# include: cuda
CUDA_INCLUD = -I/usr/local/cuda/include
CUDA_LIB = -L/usr/local/cuda/lib64
# options: nvcc
NVCC = nvcc
NVCC_OPTS_INCLUDES = $(CUDA_INCLUDE)
NVCC_ARCH = -gencode arch=compute_50,code=sm_50
NVCC_OPTS_LIBS_LD = $(CUDA_LIB) -lcublas

# g++
CC = g++
CC_FLAGS = -std=c++11

# ----------------------------------------------------------------------------

all:
	$(CC) -o $(BUILD_DIR)/test_gaussian test_gaussian.cc $(SOURCES) $(HEADERS) $(CC_FLAGS)
	$(BUILD_DIR)/test_gaussian > log
