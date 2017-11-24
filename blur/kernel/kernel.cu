#include <stdio.h>
#include "kernel.hh"
#include "gaussian.cu"

void GaussianBlurCaller(const int center, float * kernel)
{
	GaussianBlurKernel<<<1, 1>>>(center, kernel);
}
