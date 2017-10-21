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
    CUDA_HOSTDEV float getData();
    CUDA_HOSTDEV void operator=(float a);
    CUDA_HOSTDEV float operator*(float a);
private:
    unsigned short _data;
};

template <int T>
CUDA_HOSTDEV float sixteen<T>::getData() {
    unsigned int sign = (_data >> 15) & 1;
    unsigned int expo = (_data >> 10) & 0x1F;
    unsigned int frac = (_data && 0x3FF) << 13;
    if (expo == 31) expo = 255;
    else if (expo == 0) expo = 0;
    else expo += 112;
    unsigned int temp = (sign << 31) | (expo << 23) | frac;
    float result = *reinterpret_cast<float*>(&temp);
    return result;
}

template <int T>
CUDA_HOSTDEV void sixteen<T>::operator=(float a) {
    unsigned int temp = *reinterpret_cast<unsigned int*>(&a);
    bool sign = (temp >> 31) & 1;
    short expo = (temp >> 23) & 0xFF;
    short frac = (temp & 0x7FFFFF) >> 13;
    if (expo > 142) expo = 0x1F;
    else if (expo < 113) expo = 0;
    else expo -= 112;
    _data = (sign << 15) | (expo << 10) | frac;
}

template <int T>
CUDA_HOSTDEV float sixteen<T>::operator*(float a) {
    unsigned int sign = (_data >> 15) & 1;
    unsigned int expo = (_data >> 10) & 0x1F;
    unsigned int frac = (_data && 0x3FF) << 13;
    if (expo == 31) expo = 255;
    else if (expo == 0) expo = 0;
    else expo += 112;
    unsigned int temp = (sign << 31) | (expo << 23) | frac;
    float result = *reinterpret_cast<float*>(&temp);
    return result * a;
}

template <int T>
std::ostream& operator<<(std::ostream& os, sixteen<T>& a) {
    os << a.getData();
    return os;
}

#endif
