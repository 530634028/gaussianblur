#include <stdio.h>

template <typename T>
__global__ void GaussianBlurKernel(const T * img, T * dst, const int kw, const int center, float * kernel)
{
	printf("%d\f", center);
}


template <typename T>
void GaussianBlurInit(const T * img, T * dst, const int kw, const int center, float * kernel)
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
