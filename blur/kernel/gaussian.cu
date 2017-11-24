#include <stdio.h>
#include "kernel.hh"

__global__ void GaussianBlurKernel(const int center, float * kernel)
{
	printf("%d\f", center);
}
