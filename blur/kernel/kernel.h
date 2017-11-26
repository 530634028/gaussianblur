#ifndef KERNEL_H
#define KERNEL_H

template <typename T>
void GaussianBlurCaller(const T * img, T * dst, const int kw, const int center, float * kernel) {
	GaussianBlurInit(img, dst, kw, center, kernel);
}


#endif
