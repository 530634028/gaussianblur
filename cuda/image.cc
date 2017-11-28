#include <opencv2/opencv.hpp>
#include <cuda.h>
#include <cuda_runtime.h>

#include "utils.hh"
#include "image.hh"

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
}
