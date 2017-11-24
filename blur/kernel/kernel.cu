#include <stdio.h>
#include "kernel.hh"
#include "gaussian.cu"

template <typename T>
void GaussianBlurCaller(const T * img, T * dst, const int kw, const int center, float * kernel)
{
	T * d_img;
	T * d_dst;
	float * d_kernel;
	cudaMalloc(&d_img, sizeof(T));
	cudaMalloc(&d_dst, sizeof(T));
	cudaMalloc(&d_kernel, kw*sizeof(float));
	cudaMemcpy(d_img, img, sizeof(T), cudaMemcpyHostToDevice);
	cudaMemcpy(d_dst, dst, sizeof(T), cudaMemcpyHostToDevice);
	cudaMemcpy(d_kernel, kernel, kw*sizeof(float), cudaMemcpyHostToDevice);

	GaussianBlurKernel<<<1, 1>>>(d_img, d_dst, kw, center, d_kernel);

	cudaFree(d_img);
	cudaFree(d_dst);
	cudaFree(d_kernel);
}
