#include <iostream>
#include "../utils/image.hh"
#include "../utils/border.hh"

int main()
{
	int height = 3;
	int width = 4;
	int channels = 1;

	Mat<float> img(height, width, channels);
	image_init(img);
	std::cout << "image: " << std::endl;
	image_print(img);

	const int center = 2;
	const int h_height = height + 2 * center;
	const int h_width = width + 2 * center;
	float * h_img = (float*)malloc(sizeof(float)*h_height*h_width);
	border_padding(h_img, img.ptr(0), width, height, center);
	std::cout << "image padded: " << std::endl;
	for (int i = 0; i < h_height; ++i)
	{
		for (int j = 0; j < h_width; ++j)
		{
			std::cout << h_img[i * h_width + j] << ", ";
		}
		std::cout << std::endl;
	}
	free(h_img);
}
