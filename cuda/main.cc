#include <stdlib.h>
#include <string>
#include <iostream>

#include "image.hh"

void preprocessing (std::string input_file)
{
	unsigned char * h_inputImageRGBA, * d_inputImageRGBA;
	unsigned char * h_outputImageRGBA, * d_outputImageRGBA;
	unsigned char * d_redBlurred, * d_greenBlurred, * d_blueBlurred;
	float * h_filter;
	int filterWidth;

	image_preprocessing(&h_inputImageRGBA,
                            &h_outputImageRGBA,
                            &d_inputImageRGBA,
                            &d_outputImageRGBA,
                            &d_redBlurred,
                            &d_greenBlurred,
                            &d_blueBlurred,
                            &h_filter,
                            &filterWidth,
                            input_file);
}


int main (int argc, char **argv)
{
	double perPixelError = 0.0;
	double globalError = 0.0;
	bool useEpsCheck = false;

	std::string input_file;
	std::string output_file;
	std::string reference_file;
	switch (argc)
	{
		case 2:
			input_file = std::string(argv[1]);
			output_file = "_build/out.png";
			reference_file = "images/reference.png";
			break;
		case 3:
			input_file = std::string(argv[1]);
			output_file = std::string(argv[2]);
			reference_file = "images/reference.png";
			break;
		case 4:
			input_file = std::string(argv[1]);
			output_file = std::string(argv[2]);
			reference_file = std::string(argv[3]);
			break;
		case 6:
			input_file = std::string(argv[1]);
			output_file = std::string(argv[2]);
			reference_file = std::string(argv[3]);
			perPixelError = atof(argv[4]);
			globalError = atof(argv[5]);
			useEpsCheck = true;
			break;
		default:
			std::cerr << "Usage: ./main <in_file> <out_file> <ref_file> <pixelerr> <globalerr> <useeps>" << std::endl;
			exit(1);	
	}

	preprocessing(input_file);
}
