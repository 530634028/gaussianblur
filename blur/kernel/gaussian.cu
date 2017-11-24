#include <stdio.h>

extern "C"
__global__ void GaussianBlurKernel(const int center, float * kernel)
{
	printf("%d\f", center);
}
