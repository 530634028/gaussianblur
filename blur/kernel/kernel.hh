#ifndef KERNEL_HH
#define KERNEL_HH

template <typename T>
void GaussianBlurCaller(const T * img, T * dst, const int kw, const int center, float * kernel);

#endif
