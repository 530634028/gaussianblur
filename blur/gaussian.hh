#include <iostream>
#include <memory>
#include <vector>
#include <pthread.h>
#include "utils/mat.h"
#include "utils/utils.hh"
#include "utils/border.hh"
#include "gaussian.cuh"

class GaussCache {
	public:
		std::unique_ptr<float, std::default_delete<float[]>> kernel_buf;
		float* kernel;
		int kw;
		GaussCache(float sigma);
};

class GaussCacheFull {
	public:
		std::unique_ptr<float, std::default_delete<float[]>> kernel_buf;
		float* kernel;
		int kw;
		GaussCacheFull(float sigma);
};

/* ---------------------------------------------------------------------------------------- */

class GaussianBlur {
	float sigma;
	GaussCache gcache;
	public:
		GaussianBlur(float sigma): sigma(sigma), gcache(sigma) {}

		// TODO faster convolution
		template <typename T>
		Mat<T> blur(const Mat<T>& img) const {
			m_assert(img.channels() == 1);
			const int w = img.width(), h = img.height();
			Mat<T> ret(h, w, img.channels());

			const int kw = gcache.kw;
			const int center = kw / 2;
			float * kernel = gcache.kernel;

			std::vector<T> cur_line_mem(center * 2 + std::max(w, h), 0);
			T *cur_line = cur_line_mem.data() + center;

			// apply to columns
			REP(j, w){
				const T* src = img.ptr(0, j);
				// copy a column of src
				REP(i, h) {
					cur_line[i] = *src;
					src += w;
				}

				// pad the border with border value
				T v0 = cur_line[0];
				for (int i = 1; i <= center; i ++)
					cur_line[-i] = v0;
				v0 = cur_line[h - 1];
				for (int i = 0; i < center; i ++)
					cur_line[h + i] = v0;

				// sum: image[index] * kernel[k]
				T *dest = ret.ptr(0, j);
				REP(i, h) {
					T tmp{0};
					for (int k = -center; k <= center; k ++)
						tmp += cur_line[i + k] * kernel[k];
					*dest = tmp;
					dest += w;
				}
			}

			// apply to rows
			REP(i, h) {
				T *dest = ret.ptr(i);
				memcpy(cur_line, dest, sizeof(T) * w);
				{	// pad the border
					T v0 = cur_line[0];
					for (int j = 1; j <= center; j ++)
						cur_line[-j] = v0;
					v0 = cur_line[w - 1];
					for (int j = 0; j < center; j ++)
						cur_line[w + j] = v0;
				}
				// sum: image[index] * kernel[k]
				REP(j, w) {
					T tmp{0};
					for (int k = -center; k <= center; k ++)
						tmp += cur_line[j + k] * kernel[k];
					*(dest ++) = tmp;
				}
			}
			return ret;
		}
};

class MultiScaleGaussianBlur {
	std::vector<GaussianBlur> gauss;		// size = nscale - 1
	public:
	MultiScaleGaussianBlur(
			int nscale, float gauss_sigma,
			float scale_factor) {
		REP(k, nscale - 1) {
			gauss.emplace_back(gauss_sigma);
			gauss_sigma *= scale_factor;
		}
	}

	Mat32f blur(const Mat32f& img, int n) const
	{ return gauss[n - 1].blur(img); }
};

/* ---------------------------------------------------------------------------------------- */

class GaussianBlurMP {
	float sigma;
	GaussCacheFull gcache;
	public:
		GaussianBlurMP(float sigma): sigma(sigma), gcache(sigma) {}

		// TODO faster convolution
		template <typename T>
		Mat<T> blur(const Mat<T>& img) const {
			m_assert(img.channels() == 1);
			const int w = img.width(), h = img.height();
			Mat<T> ret(h, w, img.channels());

			const int kw = gcache.kw;
			const int center = kw / 2;

			T * h_img = (T*)malloc(sizeof(T)*(w+2*center)*(h+2*center));
			border_padding(h_img, img.ptr(0), w, h, center);
			const int h_img_width = w + 2 * center;
			const int h_img_height = h + 2 * center;

#pragma omp parallel for schedule(dynamic)
			for (int i = 0; i < w*h; ++i) {
				const int dst_width = i % w;
				const int dst_height = (i / w) % h;
				const int n = i / w / h;

				int hstart = dst_height;
				int wstart = dst_width;
				int hend = std::min(hstart + kw, h_img_height);
				int wend = std::min(wstart + kw, h_img_width);
				hstart = std::max(hstart, 0);
				wstart = std::max(wstart, 0);
				hend = std::min(hend, h_img_height);
				wend = std::min(wend, h_img_width);
				T tmp = 0;
				int counter = 0;
				const T * in_slice = h_img + n * h_img_height * h_img_width;
				for (int h = hstart; h < hend; ++h) {
					for (int w = wstart; w < wend; ++w) {
						tmp += in_slice[h * h_img_width + w] * gcache.kernel_buf.get()[counter];
						counter++;
					}
				}
				ret.data()[i] = tmp;
			} 

			return ret;
		}
};

/* ---------------------------------------------------------------------------------------- */

/*
template <typename T>
struct gaussian_data {
	T * h_img;
	T * ret;
	int w;
	int h;
	float * kernel;
	int kw;
	int center;
};

template <typename T>
void gaussian_pthread(struct gaussian_data<T> data) 
{
	const int h_img_width = data.w + 2 * data.center;
	const int h_img_height = data.h + 2 * data.center;

	for (int i = 0; i < data.w*data.h; ++i) {
		const int dst_width = i % data.w;
		const int dst_height = (i / data.w) % data.h;
		const int n = i / data.w / data.h;

		int hstart = dst_height;
		int wstart = dst_width;
		int hend = std::min(hstart + data.kw, h_img_height);
		int wend = std::min(wstart + data.kw, h_img_width);
		hstart = std::max(hstart, 0);
		wstart = std::max(wstart, 0);
		hend = std::min(hend, h_img_height);
		wend = std::min(wend, h_img_width);
		T tmp = 0;
		int counter = 0;
		const T * in_slice = data.h_img + n * h_img_height * h_img_width;
		for (int h = hstart; h < hend; ++h) {
			for (int w = wstart; w < wend; ++w) {
				tmp += in_slice[h * h_img_width + w] * data.kernel[counter];
				counter++;
			}
		}
		data.ret[i] = tmp;
	}
}


template <typename T>
pthread_t pthread_create(int n, struct gaussian_data<T> * data) {
	pthread_t thread_ids[n];
	for (int i = 0; i < n; ++i)
		pthread_create(&thread_ids[i], NULL, gaussian_pthread, data);

	return thread_ids;
}

void pthread_run(int n, pthread_t thread_ids[]) {
	for (int i = 0; i < n; ++i)
		pthread_join(thread_ids[i], NULL);
}
*/

class GaussianBlurPThread {
	float sigma;
	GaussCacheFull gcache;
	public:
		GaussianBlurPThread(float sigma): sigma(sigma), gcache(sigma) {}

		// TODO faster convolution
		template <typename T>
		Mat<T> blur(const Mat<T>& img) const {
			m_assert(img.channels() == 1);
			const int w = img.width(), h = img.height();
			Mat<T> ret(h, w, img.channels());

			const int kw = gcache.kw;
			const int center = kw / 2;

			T * h_img = (T*)malloc(sizeof(T)*(w+2*center)*(h+2*center));
			border_padding(h_img, img.ptr(0), w, h, center);
/*
			struct gaussian_data<T> * data = (struct gaussian_data<T>)malloc(sizeof(struct gaussian_data<T>));
			data->h_img = h_img;
			data->ret = ret.ptr(0);
			data->w = w;
			data->h = h;
			data->kernel = gcache.kernel_buf.get();
			data->kw = kw;
			data->center = center;
			int n = w * h;
			pthread_t thread_ids = pthread_create(n, data);
			pthread_run(n, thread_ids);
*/
			return ret;
		}
};


/* ---------------------------------------------------------------------------------------- */

class GaussianBlurGPU {
	float sigma;
	GaussCacheFull gcache;
	public:
		GaussianBlurGPU(float sigma): sigma(sigma), gcache(sigma) {}

		template <typename T>
		Mat<T> blur(const Mat<T>& img) const {
			m_assert(img.channels == 1);
			const int w = img.width(), h = img.height();
			Mat<T> ret(h, w, img.channels());

			const int kw = gcache.kw;
			const int center = kw / 2;

			GaussianBlurCaller(img.ptr(0), ret.ptr(0), w, h,
                                           kw, center, gcache.kernel_buf.get());

			return ret;
		}
};
