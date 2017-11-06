#ifndef FIXED_POINT_
#define FIXED_POINT_
#include <math.h>
#include <iostream>

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

class fixedp {
public:
    CUDA_HOSTDEV fixedp(int bw, int iw) { _total = bw; _integer = iw; _mentissa = bw - iw; }
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

#endif
