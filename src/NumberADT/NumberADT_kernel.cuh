// Copied from NumberADT/NumberADT.cu
// ===================================================================

CUDA_HOSTDEV float mul_float(float_t lhs, float rhs){
  return lhs * rhs;
}

CUDA_HOSTDEV float mul_half(sixteen<1> lhs, float rhs){  
  return lhs * rhs;
}


template <unsigned int BW>
CUDA_HOSTDEV float mul_exp(log2quant<BW> lhs, float rhs_native){
    return lhs * rhs_native;
}

template <unsigned int BW>
CUDA_HOSTDEV float mul_fixed(fixedp<BW, __MAX_IW__> lhs, float rhs){
  return lhs * rhs;
}

CUDA_HOSTDEV float Number::operator*(float rhs) {
  switch (_type){
    case EXP:
      if (_bwTotal == 1) return mul_exp<1>(buf_exp1, rhs);
      else if (_bwTotal == 2) return mul_exp<2>(buf_exp2, rhs);
      else if (_bwTotal == 3) return mul_exp<3>(buf_exp3, rhs);
      else if (_bwTotal == 4) return mul_exp<4>(buf_exp4, rhs);
      else if (_bwTotal == 5) return mul_exp<5>(buf_exp5, rhs);
      else if (_bwTotal == 6) return mul_exp<6>(buf_exp6, rhs);
      else if (_bwTotal == 7) return mul_exp<7>(buf_exp7, rhs);
      else if (_bwTotal == 8) return mul_exp<8>(buf_exp8, rhs);
      else if (_bwTotal == 9) return mul_exp<9>(buf_exp9, rhs);
      else if (_bwTotal == 16) return mul_exp<16>(buf_exp, rhs);
      else {
        // LOG(ERROR) << "Number::operator*(), Not a valid bitwidth "
        //            << __RED__ << _bwTotal << __END__;
        exit(-1);
      }
    case FIXED: 
      if (_bwTotal == 1) return mul_fixed<1>(buf_fixed1, rhs);
      else if (_bwTotal == 2) return mul_fixed<2>(buf_fixed2, rhs);
      else if (_bwTotal == 3) return mul_fixed<3>(buf_fixed3, rhs);
      else if (_bwTotal == 4) return mul_fixed<4>(buf_fixed4, rhs);
      else if (_bwTotal == 5) return mul_fixed<5>(buf_fixed5, rhs);
      else if (_bwTotal == 6) return mul_fixed<6>(buf_fixed6, rhs);
      else if (_bwTotal == 7) return mul_fixed<7>(buf_fixed7, rhs);
      else if (_bwTotal == 8) return mul_fixed<8>(buf_fixed8, rhs);
      else if (_bwTotal == 9) return mul_fixed<9>(buf_fixed9, rhs);
      else if (_bwTotal == 10) return mul_fixed<10>(buf_fixed10, rhs);
      else if (_bwTotal == 11) return mul_fixed<11>(buf_fixed11, rhs);
      else if (_bwTotal == 12) return mul_fixed<12>(buf_fixed12, rhs);
      else if (_bwTotal == 13) return mul_fixed<13>(buf_fixed13, rhs);
      else if (_bwTotal == 14) return mul_fixed<14>(buf_fixed14, rhs);
      else if (_bwTotal == 15) return mul_fixed<15>(buf_fixed15, rhs);
      else if (_bwTotal == 16) return mul_fixed<16>(buf_fixed16, rhs);
      else if (_bwTotal == 17) return mul_fixed<17>(buf_fixed17, rhs);
      else if (_bwTotal == 18) return mul_fixed<18>(buf_fixed18, rhs);
      else if (_bwTotal == 19) return mul_fixed<19>(buf_fixed19, rhs);
      else if (_bwTotal == 20) return mul_fixed<20>(buf_fixed20, rhs);
      else if (_bwTotal == 21) return mul_fixed<21>(buf_fixed21, rhs);
      else if (_bwTotal == 22) return mul_fixed<22>(buf_fixed22, rhs);
      else if (_bwTotal == 23) return mul_fixed<23>(buf_fixed23, rhs);
      else if (_bwTotal == 24) return mul_fixed<24>(buf_fixed24, rhs);
      else if (_bwTotal == 25) return mul_fixed<25>(buf_fixed25, rhs);
      else if (_bwTotal == 26) return mul_fixed<26>(buf_fixed26, rhs);
      else if (_bwTotal == 27) return mul_fixed<27>(buf_fixed27, rhs);
      else if (_bwTotal == 28) return mul_fixed<28>(buf_fixed28, rhs);
      else if (_bwTotal == 29) return mul_fixed<29>(buf_fixed29, rhs);
      else if (_bwTotal == 30) return mul_fixed<30>(buf_fixed30, rhs);
      else if (_bwTotal == 31) return mul_fixed<31>(buf_fixed31, rhs);
      else if (_bwTotal == 32) return mul_fixed<32>(buf_fixed32, rhs);
      else{
        // LOG(ERROR) << "Number::operator*(), Not a valid bitwidth "
        //            << __RED__ << _bwTotal << __END__;
        exit(-1);
      }
    case FLOAT: return mul_float(buf_float, rhs);
    case HALF: return mul_half(buf_half, rhs);
    default:
//       LOG(ERROR) << "Number::operator*(), Invalid type for lhs";
      exit(-1);
  }
}

void Number::operator=(float rhs){
  switch (_type){
    case EXP:
      if (_bwTotal == 1) buf_exp1 = rhs;
      else if (_bwTotal == 2) buf_exp2 = rhs;
      else if (_bwTotal == 3) buf_exp3 = rhs;
      else if (_bwTotal == 4) buf_exp4 = rhs;
      else if (_bwTotal == 5) buf_exp5 = rhs;
      else if (_bwTotal == 6) buf_exp6 = rhs;
      else if (_bwTotal == 7) buf_exp7 = rhs;
      else if (_bwTotal == 8) buf_exp8 = rhs;
      else if (_bwTotal == 9) buf_exp9 = rhs;
      else if (_bwTotal == 16) buf_exp = rhs;
      else {
        // LOG(ERROR) << "Number::operator=(), Not a valid EXP bitwidth "
        //            << __RED__ << _bwTotal << __END__;
        exit(-1);
      }
    break;
    case FIXED: 
      if (_bwTotal == 1) buf_fixed1 = rhs; 
      else if (_bwTotal == 2) buf_fixed2 = rhs; 
      else if (_bwTotal == 3) buf_fixed3 = rhs; 
      else if (_bwTotal == 4) buf_fixed4 = rhs; 
      else if (_bwTotal == 5) buf_fixed5 = rhs; 
      else if (_bwTotal == 6) buf_fixed6 = rhs; 
      else if (_bwTotal == 7) buf_fixed7 = rhs; 
      else if (_bwTotal == 8) buf_fixed8 = rhs; 
      else if (_bwTotal == 9) buf_fixed9 = rhs; 
      else if (_bwTotal == 10) buf_fixed10 = rhs;
      else if (_bwTotal == 11) buf_fixed11 = rhs;
      else if (_bwTotal == 12) buf_fixed12 = rhs;
      else if (_bwTotal == 13) buf_fixed13 = rhs;
      else if (_bwTotal == 14) buf_fixed14 = rhs;
      else if (_bwTotal == 15) buf_fixed15 = rhs;
      else if (_bwTotal == 16) buf_fixed16 = rhs; 
      else if (_bwTotal == 17) buf_fixed17 = rhs;
      else if (_bwTotal == 18) buf_fixed18 = rhs; 
      else if (_bwTotal == 19) buf_fixed19 = rhs; 
      else if (_bwTotal == 20) buf_fixed20 = rhs; 
      else if (_bwTotal == 21) buf_fixed21 = rhs;
      else if (_bwTotal == 22) buf_fixed22 = rhs;
      else if (_bwTotal == 23) buf_fixed23 = rhs;
      else if (_bwTotal == 24) buf_fixed24 = rhs;
      else if (_bwTotal == 25) buf_fixed25 = rhs;
      else if (_bwTotal == 26) buf_fixed26 = rhs;
      else if (_bwTotal == 27) buf_fixed27 = rhs;
      else if (_bwTotal == 28) buf_fixed28 = rhs;
      else if (_bwTotal == 29) buf_fixed29 = rhs;
      else if (_bwTotal == 30) buf_fixed30 = rhs;
      else if (_bwTotal == 31) buf_fixed31 = rhs;
      else if (_bwTotal == 32) buf_fixed32 = rhs; 
      else{
          // LOG(ERROR) << "Number::operator=(), Not a valid FIXED bitwidth "
          //            << __RED__ << _bwTotal << __END__;
          exit(-1);
      }
      break;
    case FLOAT: buf_float = (float_t) rhs; break;
    case HALF: buf_half = (float)rhs; break;
    default:
//       LOG(ERROR) << "Number::operator*(), Invalid type for lhs";
      exit(-1);
  }
}

// TODO: Is it necessaray to support all those 9 combinations?
CUDA_HOSTDEV void Number::operator=(const Number& rhs){
    _type = rhs._type;
    _bwTotal = rhs._bwTotal;
    _bwInt = rhs._bwInt;
  switch (_type){
    case EXP  : buf_exp   = rhs.buf_exp; break;
    case FIXED: buf_fixed = rhs.buf_fixed; break;
    case FLOAT: buf_float = rhs.buf_float; break;
    case HALF:  buf_half = rhs.buf_half; break;
    default:
//       LOG(ERROR) << "Number::operator*(), Invalid type for lhs";
      exit(-1);
  }
}
