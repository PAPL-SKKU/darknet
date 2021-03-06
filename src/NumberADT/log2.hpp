#ifndef LOG_QUANT_
#define LOG_QUANT_
#include <math.h>
#include <iostream>

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

template <int T>
class log2quant {
public:
    CUDA_HOSTDEV log2quant(): _size (T) {}
    CUDA_HOSTDEV int getSize() {return _size;}
    CUDA_HOSTDEV short getData() {return _data;}
    CUDA_HOSTDEV bool getSign() {return _sign;}
    CUDA_HOSTDEV float getResult();
    CUDA_HOSTDEV void operator=(float a);
    CUDA_HOSTDEV float operator* (float a) const;
private:
    int _size;
    short _data;
    bool _sign;
};

template <int T>
CUDA_HOSTDEV float log2quant<T>::operator*(float a) const {
    if (!a) return 0;
    int data = (_data + 128 - (1<<(T-2))) << 23;
    unsigned int temp = *reinterpret_cast<int*>(&a);
    int sign = (temp >> 31) & 0x1;
    int result = data + ((temp << 1) >> 1) - (127 << 23);
    return (*reinterpret_cast<float*>(&result)) * (1 - 2 * (_sign ^ sign));
}

template <int T>
CUDA_HOSTDEV void log2quant<T>::operator=(float a) {
    unsigned int temp = *reinterpret_cast<unsigned int*>(&a);
    _data = ((temp << 1) >> 24) + (1<<(T-2)) - 128;
    _sign = (temp >> 31) & 0x1;
}

template <int T>
CUDA_HOSTDEV float log2quant<T>::getResult() {
    int temp = (_data - (1<<T-2) + 128) << 23;
    float result = (1 - 2 * _sign) * (*reinterpret_cast<float*>(&temp));
    return result;
}

template <int T>
std::ostream& operator<<(std::ostream& os, log2quant<T>& a) {
    float result = a.getResult();
    os << result;
    return os;
}
#endif
