#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#define max(a, b) ((a) > (b) ? a : b)
#define min(a, b) ((a) < (b) ? a : b)


__global__ void GaussianBlurKernel(const float * img, float * dst, const int width, const int height,
                                   const int kw, const int center, float * kernel) 
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (width*height); i += blockDim.x * gridDim.x)
	{
		const int pw = i % kw;
		const int ph = (i / kw) % kw;
		const int n = i / kw / kw;
		int hstart = ph - center;
		int wstart = pw - center;
		int hend = min(hstart + kw, height + center);
		int wend = min(wstart + kw, width + center);
		hstart = max(hstart, 0);
		wstart = max(wstart, 0);
		hend = min(hend, height);
		wend = min(wend, width);

		float val = 0;
		const float * in_slice = img + n * height * width;
		int counter = 0;
		for (int h = hstart; h < hend; ++h)
		{
			for (int w = wstart; w < wend; ++w)
			{
				val += in_slice[h * width + w] * kernel[counter];
				counter++;
			}
		}
		dst[i] = val;
	}
}


void GaussianBlurCaller(const float * img, float * dst, const int w, const int h,
                        const int kw, const int center, float * kernel) 
{
	float * d_img;
	float * d_dst;
	float * d_kernel;
	cudaMalloc(&d_img, sizeof(float)*w*h);
	cudaMalloc(&d_dst, sizeof(float)*w*h);
	cudaMalloc(&d_kernel, sizeof(float)*kw*kw);
	cudaMemcpy(d_img, img, sizeof(float)*w*h, cudaMemcpyHostToDevice);
	cudaMemcpy(d_dst, dst, sizeof(float)*w*h, cudaMemcpyHostToDevice);
	cudaMemcpy(d_kernel, kernel, sizeof(float)*kw*kw, cudaMemcpyHostToDevice);

	int threadsPerBlock = 512;
	int numBlocks = (w * h + threadsPerBlock - 1) / threadsPerBlock;
	GaussianBlurKernel<<<numBlocks, threadsPerBlock>>>(d_img, d_dst, w, h, kw, center, d_kernel);
	cudaMemcpy(dst, d_dst, sizeof(float)*w*h, cudaMemcpyDeviceToHost);

	cudaFree(d_img);
	cudaFree(d_dst);
	cudaFree(d_kernel);
}
