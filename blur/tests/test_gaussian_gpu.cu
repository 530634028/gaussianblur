#include "../gaussian.cu"


int main()
{
	const int kw = 5;
	const int w = 3;
	const int h = 4;
	const int center = kw / 2;
	GaussianBlurKernel<<<1, 1>>>(center, w, h);
	return 0;
}
