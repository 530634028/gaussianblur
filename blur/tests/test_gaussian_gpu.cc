#include "../utils/image.hh"
//#include "../gaussian.cu"


template <typename T>
void test_gaussian_gpu(Mat<T> img) 
{
	//GaussianBlur blur(2);
	//Mat<T> img_blurred = blur.blur(img);

	//std::cout << "image_blurred: " << std::endl;
	//image_print(img_blurred);
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

	test_gaussian_gpu(img);
}

