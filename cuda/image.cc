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

	cv::cvtColor(image, imageInputRGBA, CV_BGR2BGRA);
	imageOutputRGBA.create(image.rows, image.cols, CV_8UC4);
	if (!imageInputRGBA.isContinuous() || !imageOutputRGBA.isContinuous())
	{
		std::cerr << "Images aren't continuous. Exiting." << std::endl;
		exit(1);
	}

	* h_inputImageRGBA = (unsigned char*)imageInputRGBA.ptr<unsigned char>(0);
	* h_outputImageRGBA = (unsigned char*)imageOutputRGBA.ptr<unsigned char>(0);
	const size_t numPixels = imageInputRGBA.rows * imageInputRGBA.cols;
	checkCudaErrors(cudaMalloc(d_inputImageRGBA, sizeof(unsigned char)*numPixels));
	checkCudaErrors(cudaMalloc(d_outputImageRGBA, sizeof(unsigned char)*numPixels));
	checkCudaErrors(cudaMemset(*d_outputImageRGBA, 0, sizeof(unsigned char)*numPixels));
	checkCudaErrors(cudaMemcpy(*d_inputImageRGBA, *h_inputImageRGBA, sizeof(unsigned char)*numPixels, cudaMemcpyHostToDevice));

	unsigned char * d_inputImageRGBA__ = * d_inputImageRGBA;
	unsigned char * d_outputImageRGBA__ = * d_outputImageRGBA;
	const int blurKernelWidth = 9;
	const float blurKernelSigma = 2.;
	* filterWidth = blurKernelWidth;
	* h_filter = new float[blurKernelWidth * blurKernelWidth];
	float * h_filter__ = * h_filter;
	float filterSum = 0.f;
	for (int r = -blurKernelWidth/2; r <= blurKernelWidth/2; ++r)
	{
		for (int c = -blurKernelWidth/2; c <= blurKernelWidth/2; ++c)
		{
			float filterValue = expf(-(float)(c*c + r*r)/(2.f*blurKernelSigma*blurKernelSigma));
			(*h_filter)[(r+blurKernelWidth/2) * blurKernelWidth + c + blurKernelWidth/2] = filterValue;
			filterSum += filterValue;
		}
	}
	float normalizationFactor = 1.f / filterSum;
	for (int r = -blurKernelWidth/2; r <= blurKernelWidth/2; ++r)
	{
		for (int c = -blurKernelWidth/2; c <= blurKernelWidth/2; ++c)
		{
			(*h_filter)[(r+blurKernelWidth/2) * blurKernelWidth + c + blurKernelWidth/2] *= normalizationFactor;
		}
	}

	checkCudaErrors(cudaMalloc(d_redBlurred, sizeof(unsigned char)*numPixels));
	checkCudaErrors(cudaMalloc(d_greenBlurred, sizeof(unsigned char)*numPixels));
	checkCudaErrors(cudaMalloc(d_blueBlurred, sizeof(unsigned char)*numPixels));
	checkCudaErrors(cudaMemset(*d_redBlurred, 0, sizeof(unsigned char)*numPixels));
	checkCudaErrors(cudaMemset(*d_greenBlurred, 0, sizeof(unsigned char)*numPixels));
	checkCudaErrors(cudaMemset(*d_blueBlurred, 0, sizeof(unsigned char)*numPixels));
}
