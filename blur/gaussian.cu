#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#define max(a, b) ((a) > (b) ? a : b)


__global__ void GaussianBlurKernel(const float * img, float * dst, const int w, const int h,
                                   const int kw, const int center, float * kernel) 
{
	float * d_cur_line_mem;
	cudaMalloc(&d_cur_line_mem, sizeof(float)*(2 * center + max(w, h)));
	float * d_cur_line = d_cur_line_mem + center;

	for (int j = 0; j < w; ++j)
	{
		const float * src = img + j;
		for (int i = 0; i < h; ++i)
		{
			d_cur_line[i] = * src;
			src += w;
		}

		float v0 = d_cur_line[0];
		for (int i = 1; i <= center; ++i)
			d_cur_line[-i] = v0;
		v0 = d_cur_line[h - 1];
		for (int i = 0; i < center; ++i)
			d_cur_line[h + i] = v0;

		float * dest = dst + j;
		for (int i = 0; i < h; ++i)
		{
			float tmp = 0;
			for (int k = -center; k <= center; ++k)
			{
				tmp += d_cur_line[i + k] * kernel[k];
			}
			*dest = tmp;
			dest += w;
		}
	}

	for (int i = 0; i < h; ++i)
	{
		float * dest = dst + i * w;
		for (int j = 0; j < w; ++j)
			d_cur_line[j] = dest[j];
		float v0 = d_cur_line[0];
		for (int j = 1; j <= center; ++j)
			d_cur_line[-j] = v0;
		v0 = d_cur_line[w - 1];
		for (int j = 0; j < center; ++j)
			d_cur_line[w + j] = v0;

		for (int j = 0; j < w; ++j)
		{
			float tmp = 0;
			for (int k = -center; k <= center; ++k)
			{
				tmp += d_cur_line[j + k] * kernel[k];
			}
			*(dest++) = tmp;
		}	
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
	cudaMalloc(&d_kernel, sizeof(float)*kw);
	cudaMemcpy(d_img, img, sizeof(float)*w*h, cudaMemcpyHostToDevice);
	cudaMemcpy(d_dst, dst, sizeof(float)*w*h, cudaMemcpyHostToDevice);
	cudaMemcpy(d_kernel, kernel, sizeof(float)*kw, cudaMemcpyHostToDevice);

	GaussianBlurKernel<<<1, 1>>>(d_img, d_dst, w, h, kw, center, d_kernel);
	cudaMemcpy(dst, d_dst, sizeof(float)*w*h, cudaMemcpyDeviceToHost);
	for (int i = 0; i < w*h; ++i) std::cout << dst[i] << ", ";

	cudaFree(d_img);
	cudaFree(d_dst);
	cudaFree(d_kernel);
}
