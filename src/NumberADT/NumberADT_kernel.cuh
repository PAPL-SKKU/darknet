// Copied from NumberADT/NumberADT.cu
// ===================================================================
#include <stdio.h>

CUDA_HOSTDEV float mul_float(float lhs, float rhs){
    return lhs * rhs;
}

CUDA_HOSTDEV float mul_half(float lhs, float rhs){
    sixteen<1> temp;
    temp = lhs;
    return temp * rhs;
}

CUDA_HOSTDEV float mul_exp(float lhs, float rhs_native, int BW){
    log2quant temp(BW);
    temp = lhs;
    return temp * rhs_native;
}

CUDA_HOSTDEV float mul_fixed(float lhs, float rhs, int BW, int IW){
    fixedp temp(BW, IW);
    temp = lhs;
    return temp * rhs;
}

CUDA_HOSTDEV float Number::operator*(float rhs) {
  float ret;
  switch (_type){
    case LOG2:
        ret = mul_exp(buf, rhs, _bwTotal);
        break;
    case FIXED: 
        ret = mul_fixed(buf, rhs, _bwTotal, 0);
        break;
    case FLOAT: 
        ret = mul_float(buf, rhs); 
        break;
    case HALF:  
        ret = mul_half(buf, rhs); 
        break;
    default:
      printf("Number::operator*(float), invalid type");
      exit(-1);
  }
  return ret;
}

void Number::operator=(float rhs){
    buf = rhs;
}

// TODO: Is it necessaray to support all those 9 combinations?
CUDA_HOSTDEV void Number::operator=(const Number& rhs){
    _type = rhs._type;
    _bwTotal = rhs._bwTotal;
    _bwInt = rhs._bwInt;
    buf = rhs.buf;
}

CUDA_HOSTDEV const float fixedp::operator*(const float rhs) const {
    float lhs = (1-2*_sign)*(float)_data / exp2f((float)_total);
    return lhs * rhs;
}

CUDA_HOSTDEV void fixedp::operator=(float rhs) {
    unsigned int temp = *reinterpret_cast<unsigned int*>(&rhs);
    _sign = temp >> 31 & 0x1;
    _data = lroundf(rhs*exp2f((float)_mentissa)) << (_total - _mentissa);
    if (_sign) _data = ~_data + 1;
}

CUDA_HOSTDEV void fixedp::operator=(double rhs) {
    unsigned long temp = *reinterpret_cast<unsigned long*>(&rhs);
    _sign = temp >> 63 & 0x1;
    _data = lroundf(rhs*exp2f((float)_mentissa)) << (_total - _mentissa);
    if (_sign) _data = ~_data + 1;
}

CUDA_HOSTDEV void fixedp::operator=(const int rhs) {
    _sign = (rhs >> 31) & 0x1;
    _data = lroundf(rhs*exp2f((float)_mentissa)) << (_total - _mentissa);
    if (_sign) _data = ~_data + 1;
}

CUDA_HOSTDEV float log2quant::operator*(float a) const {
    if (!a) return 0;
    int temp = (_data - (1<<_size-2) + 128) << 23;
    float lhs = *reinterpret_cast<float*>(&temp);
    return lhs * a * (1 - 2*_sign);
}

CUDA_HOSTDEV void log2quant::operator=(float a) {
    unsigned int temp = *reinterpret_cast<unsigned int*>(&a);
    short limit = (1 << (_size-1)) - 1;
    _data = ((temp << 1) >> 24) - 128 + (1 << (_size-2));
    if (_data > limit) _data = limit;
    if (_data < 0) _data = 0;
    _sign = (temp >> 31) & 0x1;
}

CUDA_HOSTDEV float log2quant::getResult() {
    int temp = (_data - (1<<_size-2) + 128) << 23;
    float result = (1 - 2 * _sign) * (*reinterpret_cast<float*>(&temp));
    return result;
}
