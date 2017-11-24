#include <stdio.h>
#include "gaussian.cu"

extern "C"
void GaussianBlurCaller(const int center, float * kernel)
{
	GaussianBlurKernel<<<1, 1>>>(center, kernel);
}
