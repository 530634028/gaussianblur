#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include "utils/border.hh"

#define max(a, b) ((a) > (b) ? a : b)
#define min(a, b) ((a) < (b) ? a : b)


__global__ void GaussianBlurKernel(const float * img, float * dst, const int width, const int height,
                                   const int kw, const int center, float * kernel) 
{
	const int img_width = width + 2 * center;
	const int img_height = height + 2 * center;
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (width*height); i += blockDim.x * gridDim.x)
	{
		const int dst_width = i % width;
		const int dst_height = (i / width) % height;
		const int n = i / width / height;

		int hstart = dst_height;
		int wstart = dst_width;
		int hend = min(hstart + kw, img_height);
		int wend = min(wstart + kw, img_width);
		hstart = max(hstart, 0);
		wstart = max(wstart, 0);
		hend = min(hend, img_height);
		wend = min(wend, img_width);		
		float tmp = 0;
		int counter = 0;
		const float * in_slice = img + n * img_height * img_width;
		for (int h = hstart; h < hend; ++h)
		{
			for (int w = wstart; w < wend; ++w)
			{
				tmp += in_slice[h * img_width + w] * kernel[counter];
				counter++;
			}
		}
		dst[i] = tmp;
	}
}


void GaussianBlurCaller(const float * img, float * dst, const int w, const int h,
                        const int kw, const int center, float * kernel) 
{
	// padding
	float * h_img = (float*)malloc(sizeof(float)*(w+2*center)*(h+2*center));
	border_padding(h_img, img, w, h, center);

	float * d_img;
	float * d_dst;
	float * d_kernel;
	cudaMalloc(&d_img, sizeof(float)*(w+2*center)*(h+2*center));
	cudaMalloc(&d_dst, sizeof(float)*w*h);
	cudaMalloc(&d_kernel, sizeof(float)*kw*kw);
	cudaMemcpy(d_img, h_img, sizeof(float)*(w+2*center)*(h+2*center), cudaMemcpyHostToDevice);
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
