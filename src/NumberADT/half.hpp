#ifndef NUMBER_HALF_
#define NUMBER_HALF_
#include <math.h>
#include <iostream>

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif


template <int T>
class sixteen {
public:
    CUDA_HOSTDEV sixteen(): _data (T) {}
    CUDA_HOSTDEV void init();
    CUDA_HOSTDEV float getData();
    CUDA_HOSTDEV void operator=(float a);
    CUDA_HOSTDEV float operator*(const float a) const;
private:
    unsigned short _data;
};

template <int T>
CUDA_HOSTDEV void sixteen<T>::init() {
    _data = 0;
}

template <int T>
CUDA_HOSTDEV float sixteen<T>::getData() {
    unsigned int sign = (_data&0x8000)<<16;
    unsigned int exp = _data&0x7c00;
    if (exp == 0) exp = 0;
    else if (exp == 31) exp = 255;
    else exp = (exp+0x1c000)<<13;
    unsigned int frac = (_data&0x03ff)<<13;
    unsigned int temp = sign | exp;
    temp |= frac;
    float result = *reinterpret_cast<float*>(&temp);
    return result;
}

template <int T>
CUDA_HOSTDEV void sixteen<T>::operator=(float a) {
    unsigned int temp = *reinterpret_cast<unsigned int*>(&a);
    unsigned int sign = (temp>>16)&0x8000;
    unsigned int exp = (temp&0x7f800000) >> 23;
    if (exp >= 143) _data = sign | 0x7c00;
    else if (exp <= 111) _data = sign | 0x0000;
    else _data = ((temp>>16)&0x8000) | ((((temp&0x7f800000)-0x38000000)>>13)&0x7c00) | ((temp>>13)&0x03ff);
}

template <int T>
CUDA_HOSTDEV float sixteen<T>::operator*(const float a) const {
    unsigned int sign = (_data&0x8000)<<16;
    unsigned int exp = _data&0x7c00;
    if (exp == 0) exp = 0;
    else if (exp == 31) exp = 255;
    else exp = (exp+0x1c000)<<13;
    unsigned int frac = (_data&0x03ff)<<13;
    unsigned int temp = sign | exp;
    temp |= frac;
    float result = *reinterpret_cast<float*>(&temp);
    return result * a;
}

template <int T>
std::ostream& operator<<(std::ostream& os, sixteen<T>& a) {
    os << a.getData();
    return os;
}
#endif
