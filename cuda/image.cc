#include <opencv2/opencv.hpp>
#include <cuda.h>
#include <cuda_runtime.h>

#include "utils.hh"
#include "image.hh"

cv::Mat imageInputRGBA;
cv::Mat imageOutputRGBA;

void image_preprocessing (unsigned char ** h_inputImageRGBA,
                          unsigned char ** h_outputImageRGBA,
                          unsigned char ** d_inputImageRGBA,
                          unsigned char ** d_outputImageRGBA,
                          unsigned char ** d_redBlurred,
                          unsigned char ** d_greenBlurred,
                          unsigned char ** d_blueBlurred,
                          float ** h_filter,
                          int * filterWidth,
                          const std::string &filename)
{
	checkCudaErrors(cudaFree(0));
	
	cv::Mat image = cv::imread(filename.c_str(), CV_LOAD_IMAGE_COLOR);
	if (image.empty())
	{
		std::cerr << "Couldn't open file " << filename << std::endl;
		exit(1);
	}

	cv::cvColor(image, imageInputRGBA, CV_BGR2BGRA);
	imageOutputRGBA.create(image.rows, image.cols, CV_8UC4);
	if (!imageInputRGBA.isContinuous() || !imageOutputRGBA.isContinuous())
	{
		std::cerr << "Images aren't continuous. Exiting." << std::endl;
		exit(1);
	}

	* h_inputImageRGBA = (unsigned char*)imageInputRGBA.ptr<unsigned char>(0);
	* h_outputImageRGBA = (unsigned char*)imageOutputRGBA.ptr<unsigned char>(0);
}
