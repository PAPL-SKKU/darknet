// Copied from NumberADT/NumberADT.cu
// ===================================================================

CUDA_HOSTDEV float mul_float(float lhs, float rhs){
    return lhs * rhs;
}

CUDA_HOSTDEV float mul_half(float lhs, float rhs){
    sixteen<1> temp;
    temp = lhs;
    return temp * rhs;
}


template <unsigned int BW>
CUDA_HOSTDEV float mul_exp(float lhs, float rhs_native){
    log2quant<BW> temp;
    temp = lhs;
    return temp * rhs_native;
}

template <unsigned int BW>
CUDA_HOSTDEV float mul_fixed(float lhs, float rhs){
    fixedp<BW, __MAX_IW__> temp;
    temp = lhs;
    return temp * rhs;
}

CUDA_HOSTDEV float Number::operator*(float rhs) {
  switch (_type){
    case EXP:
        if (_bwTotal == 1) return mul_exp<1>(buf, rhs);
        else if (_bwTotal == 2) return mul_exp<2>(buf, rhs);
        else if (_bwTotal == 3) return mul_exp<3>(buf, rhs);
        else if (_bwTotal == 4) return mul_exp<4>(buf, rhs);
        else if (_bwTotal == 5) return mul_exp<5>(buf, rhs);
        else if (_bwTotal == 6) return mul_exp<6>(buf, rhs);
        else if (_bwTotal == 7) return mul_exp<7>(buf, rhs);
        else if (_bwTotal == 8) return mul_exp<8>(buf, rhs);
        else if (_bwTotal == 9) return mul_exp<9>(buf, rhs);
        else if (_bwTotal == 16) return mul_exp<16>(buf, rhs);
    case FIXED: 
      if (_bwTotal == 1) return mul_fixed<1>(buf, rhs);
      else if (_bwTotal == 2) return mul_fixed<2>(buf, rhs);
      else if (_bwTotal == 3) return mul_fixed<3>(buf, rhs);
      else if (_bwTotal == 4) return mul_fixed<4>(buf, rhs);
      else if (_bwTotal == 5) return mul_fixed<5>(buf, rhs);
      else if (_bwTotal == 6) return mul_fixed<6>(buf, rhs);
      else if (_bwTotal == 7) return mul_fixed<7>(buf, rhs);
      else if (_bwTotal == 8) return mul_fixed<8>(buf, rhs);
      else if (_bwTotal == 9) return mul_fixed<9>(buf, rhs);
      else if (_bwTotal == 10) return mul_fixed<10>(buf, rhs);
      else if (_bwTotal == 11) return mul_fixed<11>(buf, rhs);
      else if (_bwTotal == 12) return mul_fixed<12>(buf, rhs);
      else if (_bwTotal == 13) return mul_fixed<13>(buf, rhs);
      else if (_bwTotal == 14) return mul_fixed<14>(buf, rhs);
      else if (_bwTotal == 15) return mul_fixed<15>(buf, rhs);
      else if (_bwTotal == 16) return mul_fixed<16>(buf, rhs);
      else if (_bwTotal == 17) return mul_fixed<17>(buf, rhs);
      else if (_bwTotal == 18) return mul_fixed<18>(buf, rhs);
      else if (_bwTotal == 19) return mul_fixed<19>(buf, rhs);
      else if (_bwTotal == 20) return mul_fixed<20>(buf, rhs);
      else if (_bwTotal == 21) return mul_fixed<21>(buf, rhs);
      else if (_bwTotal == 22) return mul_fixed<22>(buf, rhs);
      else if (_bwTotal == 23) return mul_fixed<23>(buf, rhs);
      else if (_bwTotal == 24) return mul_fixed<24>(buf, rhs);
      else if (_bwTotal == 25) return mul_fixed<25>(buf, rhs);
      else if (_bwTotal == 26) return mul_fixed<26>(buf, rhs);
      else if (_bwTotal == 27) return mul_fixed<27>(buf, rhs);
      else if (_bwTotal == 28) return mul_fixed<28>(buf, rhs);
      else if (_bwTotal == 29) return mul_fixed<29>(buf, rhs);
      else if (_bwTotal == 30) return mul_fixed<30>(buf, rhs);
      else if (_bwTotal == 31) return mul_fixed<31>(buf, rhs);
      else if (_bwTotal == 32) return mul_fixed<32>(buf, rhs);
      else{
        // LOG(ERROR) << "Number::operator*(), Not a valid bitwidth "
        //            << __RED__ << _bwTotal << __END__;
        exit(-1);
      }
    case FLOAT: return mul_float(buf, rhs);
    case HALF: return mul_half(buf, rhs);
    default:
//       LOG(ERROR) << "Number::operator*(), Invalid type for lhs";
      exit(-1);
  }
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
