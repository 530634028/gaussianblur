#ifndef GAUSSIAN_CUH
#define GAUSSIAN_CUH

void GaussianBlurCaller(const float * img, float * dst, const int w, const int h,
                        const int kw, const int center, float * kernel);

#endif // GAUSSIAN_CUH
