#ifndef GAUSSIAN_CUH
#define GAUSSIAN_CUH

template <typename T>
void GaussianBlurCaller(const T * img, T * dst, const int w, const int h,
                        const int kw, const int center, float * kernel);

#endif // GAUSSIAN_CUH
