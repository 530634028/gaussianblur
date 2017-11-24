#include <iostream>

#include "../gaussian.hh"


template <typename T>
void image_init(Mat<T> img)
{
	for (int i = 0; i < img.pixels(); ++i)
		img.data()[i] = i;
}


template <typename T>
void image_print(Mat<T> img)
{
	for (int i = 0; i < img.rows(); ++i)
	{
		for (int j = 0; j < img.cols(); ++j)
			std::cout << *img.ptr(i, j) << ", ";
		std::cout << std::endl;
	}
}


template <typename T>
void test_gaussian_cpu(Mat<T> img) 
{
	GaussianBlur blur(2);
	Mat<T> img_blurred = blur.blur(img);

	std::cout << "image_blurred: " << std::endl;
	image_print(img_blurred);
}


int main()
{
	int height = 3;
	int width = 4;
	int channels = 1;

	Mat<float> img(height, width, channels);
	image_init(img);
	std::cout << "image: " << std::endl;
	image_print(img);

	test_gaussian_cpu(img);
}
