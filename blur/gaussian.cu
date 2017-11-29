#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "gaussian.cuh"

__global__ void GaussianBlurKernel(const float * img, float * dst, const int kw, const int center, float * kernel)
{
	printf("%d\n", center);
}

void GaussianBlurCaller(const float * img, float * dst, const int kw, const int center, float * kernel) {
	float * d_img;
	float * d_dst;
	float * d_kernel;
	cudaMalloc(&d_img, sizeof(float));
	cudaMalloc(&d_dst, sizeof(float));
	cudaMalloc(&d_kernel, sizeof(float)*kw);
	cudaMemcpy(d_img, img, sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_dst, dst, sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_kernel, kernel, sizeof(float)*kw, cudaMemcpyHostToDevice);

	GaussianBlurKernel<<<1, 1>>>(d_img, d_dst, kw, center, d_kernel);

	cudaFree(d_img);
	cudaFree(d_dst);
	cudaFree(d_kernel);
}
