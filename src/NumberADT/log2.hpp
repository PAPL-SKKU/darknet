#ifndef LOG_QUANT_
#define LOG_QUANT_
#include <math.h>
#include <iostream>
#include <stdio.h>

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

class log2quant {
public:
    CUDA_HOSTDEV log2quant(int BW) { _size = BW; }
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

// CUDA_HOSTDEV float log2quant::operator*(float a) const {
//     if (!a) return 0;
//     int temp = (_data - (1<<_size-2) + 128) << 23;
//     float lhs = *reinterpret_cast<float*>(&temp);
//     return lhs * a * (1 - 2*_sign);
// }
// 
// CUDA_HOSTDEV void log2quant::operator=(float a) {
//     unsigned int temp = *reinterpret_cast<unsigned int*>(&a);
//     short limit = (1 << (_size-1)) - 1;
//     _data = ((temp << 1) >> 24) - 128 + (1 << (_size-2));
//     if (_data > limit) _data = limit;
//     if (_data < 0) _data = 0;
//     _sign = (temp >> 31) & 0x1;
// }
// 
// CUDA_HOSTDEV float log2quant::getResult() {
//     int temp = (_data - (1<<_size-2) + 128) << 23;
//     float result = (1 - 2 * _sign) * (*reinterpret_cast<float*>(&temp));
//     return result;
// }
// 
// std::ostream& operator<<(std::ostream& os, log2quant& a) {
//     os << a.getResult();
//     return os;
// }
#endif
