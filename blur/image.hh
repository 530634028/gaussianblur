#pragma once
#include "mat.h"

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
