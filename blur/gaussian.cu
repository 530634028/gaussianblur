#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

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


void GaussianBlurPadding(float * h_img, const float * img, const int w, const int h, const int center)
{
	float * cur_line = (float*)malloc(sizeof(float)*w);
	float * h_img_index;
	memcpy(cur_line, img, sizeof(float)*w);
	for (int i = 0; i < center; ++i)
	{
		h_img_index = h_img + i * (w + 2 * center);
		for (int j = 0; j < center; ++j)
			*(h_img_index + j) = cur_line[0];
		memcpy(h_img_index + center, cur_line, sizeof(float)*w);
		for (int j = 0; j < center; ++j)
			*(h_img_index + center + w + j) = cur_line[w-1];
	}
	for (int i = 0; i < h; ++i)
	{
		h_img_index = h_img + (w + 2 * center) * center + i * (w + 2 * center);
		for (int j = 0; j < center; ++j)
			*(h_img_index + j) = *(img + i * w);
		memcpy(h_img_index + center, img + i * w, sizeof(float)*w);
		for (int j = 0; j < center; ++j)
			*(h_img_index + center + w +j) = *(img + i * w + w - 1);
	}
	for (int i = 0; i < center; ++i)
	{
		h_img_index = h_img + (w + 2 * center) * center + (w + 2 * center) * h + i * (w + 2 * center);
		for (int j = 0; j < center; ++j)
			*(h_img_index + j) = *(img + (h-1)*w);
		memcpy(h_img_index + center, img + (h-1)*w, sizeof(float)*w);
		for (int j = 0; j < center; ++j)
			*(h_img_index + center + w + j) = *(img + (h-1)*w + w - 1);
	}
}

void GaussianBlurCaller(const float * img, float * dst, const int w, const int h,
                        const int kw, const int center, float * kernel) 
{
	// padding
	float * h_img = (float*)malloc(sizeof(float)*(w+2*center)*(h+2*center));
	GaussianBlurPadding(h_img, img, w, h, center);

	for (int i = 0; i < (h+2*center); ++i)
	{
		for (int j = 0; j < (w+2*center); ++j)
		{
			std::cout << h_img[i*(w+2*center) + j] << ", ";
		}
		std::cout << std::endl;
	}

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
