#ifndef FIXED_POINT_
#define FIXED_POINT_
#include <math.h>
#include <iostream>

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

template <int T, int I>
class fixedp {
public:
    CUDA_HOSTDEV fixedp (): _total (T), _integer (I), _mentissa(T-I) {}
    CUDA_HOSTDEV unsigned long long getData() {return _data;}
    CUDA_HOSTDEV int getTotal() {return _total;}
    CUDA_HOSTDEV bool getSign() {return _sign;}
    CUDA_HOSTDEV const float operator* (const float a) const;
    CUDA_HOSTDEV void operator=(float a);
    CUDA_HOSTDEV void operator=(double a);
    CUDA_HOSTDEV void operator=(const int a);
private:
    int _total;
    int _integer;
    int _mentissa;
    bool _sign;
    unsigned long long _data;
};

template <int T, int I>
CUDA_HOSTDEV const float fixedp<T, I>::operator*(const float rhs) const {
    fixedp<32,16> temp;
    temp = rhs;
    int sign = _sign ^ temp.getSign();
    float result = (1-2*sign)*(float)((_data*temp.getData())>>(32+_total));
    return result;
}

template <int T, int I>
CUDA_HOSTDEV void fixedp<T, I>::operator=(float rhs) {
    unsigned int temp = *reinterpret_cast<unsigned int*>(&rhs);
    _sign = temp >> 31 & 0x1;
    _data = lroundf(rhs*exp2f((float)_mentissa)) << (_total - _mentissa);
    if (_sign) _data = ~_data + 1;
}

template <int T, int I>
CUDA_HOSTDEV void fixedp<T, I>::operator=(double rhs) {
    unsigned long temp = *reinterpret_cast<unsigned long*>(&rhs);
    _sign = temp >> 63 & 0x1;
    _data = lroundf(rhs*exp2f((float)_mentissa)) << (_total - _mentissa);
    if (_sign) _data = ~_data + 1;
}

template <int T, int I>
CUDA_HOSTDEV void fixedp<T, I>::operator=(const int rhs) {
    _sign = (rhs >> 31) & 0x1;
    _data = lroundf(rhs*exp2f((float)_mentissa)) << (_total - _mentissa);
    if (_sign) _data = ~_data + 1;
}

template <int T, int I>
std::ostream& operator<<(std::ostream& os, fixedp<T, I>& fp) {
    os << (1-2*fp.getSign())*(float)fp.getData() / exp2f((float)fp.getTotal());
    return os;
}

#endif
