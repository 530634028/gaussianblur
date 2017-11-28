BUILD_DIR = _build

# --------------------------------------------------------------------------------------

OPENCV_INCLUDE_DIR = /usr/include
OPENCV_LIB_DIR = /usr/lib
OPENCV_FLAGS = -I$(OPENCV_INCLUDE_DIR)
OPENCV_LIBS = -L$(OPENCV_LIB_DIR) -lopencv_core -lopencv_imgproc -lopencv_highgui

CUDA_INCLUDE_DIR = /usr/local/cuda/include
CUDA_LIB_DIR = /usr/local/cuda/lib64 
CUDA_FLAGS = -I$(CUDA_INCLUDE_DIR)
CUDA_LIBS = -L$(CUDA_LIB_DIR) -lcuda -lcudart

# --------------------------------------------------------------------------------------

CC = g++
CC_OPTS = -O3 -Wall -Wextra -m64
CC_FLAGS = $(CC_OPTS) $(OPENCV_FLAGS)
CC_LDFLAGS = $(OPENCV_LIBS)

NVCC = nvcc
NVCC_OPTS = -O3 -arch=sm_50 -Xcompiler -Wall -Xcompiler -Wextra -m64
NVCC_FLAGS = $(NVCC_OPTS) -I$(CUDA_INCLUDE_DIR)
NVCC_LDFLAGS = -L$(CUDA_LIB_DIR) -lcuda -lcudart

# --------------------------------------------------------------------------------------

all: image
	$(CC) -o $(BUILD_DIR)/main main.cc $(BUILD_DIR)/image.o -I./ $(CC_FLAGS) $(CC_LDFLAGS) $(CUDA_LIBS) 
	$(BUILD_DIR)/main images/cinque_terre_small.jpg $(BUILD_DIR)/out.jpg

image:
	$(NVCC) -o $(BUILD_DIR)/image.o -c image.cc -I./ $(NVCC_FLAGS) $(NVCC_LDFLAGS) $(OPENCV_FLAGS) $(OPENCV_LIBS)

clean:
	rm -rf $(BUILD_DIR)/*
