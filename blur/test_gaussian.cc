#include <iostream>

#include "gaussian.hh"


template <typename T>
void image_print(Mat<T> img)
{
	for (int i = 0; i < img.rows(); ++i)
	{
		for (int j = 0; j < img.cols(); ++j)
		{
			std::cout << *img.ptr(i, j) << ", ";
		}
		std::cout << std::endl;
	}
}


template <typename T>
void test_gaussian_cpu(Mat<T> img, Mat<T> img_blurred) 
{
	GaussianBlur blur(2);
	img_blurred = blur.blur(img);
}


template <typename T>
void test_gaussian_gpu(Mat<T> img, Mat<T> img_blurred)
{
	GaussianBlurGPU blur(2);
	img_blurred = blur.blur(img);
}


int main()
{
	int height = 3;
	int width = 4;
	int channels = 1;

	Mat<float> img(height, width, channels);
	for (int i = 0; i < img.pixels(); ++i)
		img.data()[i] = i;
	std::cout << "image: " << std::endl;
	image_print(img);

	Mat<float> img_blurred_cpu(height, width, channels);
	test_gaussian_cpu(img, img_blurred_cpu);
	std::cout << "image_blurred_cpu: " << std::endl;
	image_print(img_blurred_cpu);

	Mat<float> img_blurred_gpu(height, width, channels);
	test_gaussian_gpu(img, img_blurred_gpu);
	std::cout << "image_blurred_gpu: " << std::endl;
	image_print(img_blurred_gpu);
}
