#include <stdio.h>

template <typename T>
__global__ void GaussianBlurKernel(const T * img, T * dst, const int kw, const int center, float * kernel)
{
	printf("%d\f", center);
}
