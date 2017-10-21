#ifdef OPENDNN

extern "C" {
#include "opendnn.h"
}
#include <string>
#include <sstream>
#include "NumberADT/Number.hpp"


// Extracted from NumberADT/Number.cu
// 

__device__ float mul_float(float_t lhs, float rhs){
  return lhs * rhs;
}

__device__ float mul_half(sixteen<1> lhs, float rhs){  
  return lhs * rhs;//__float2half(__half2float(lhs) * rhs);
}


template <unsigned int BW>
__device__ float mul_exp(log2quant<BW> lhs, float rhs_native){
//   const unsigned int BIAS = exp2f((float)BW-2) - 1;
//   const unsigned int EXP_MAX = exp2f((float)BW-1) - 1;
//
//   ap_uint<32> rhs = *(reinterpret_cast<unsigned int*>(&rhs_native));
//
//   ap_uint<1> lhs_sign = lhs[BW-1]; // sign from exp_t
//   ap_uint<1> rhs_sign = rhs[31];   // sign from float
//
//   ap_uint<BWOpr> lhs_exp = lhs.range(BW-2,0);  // exp from uint
//   ap_uint<BWOpr> rhs_exp = rhs.range(30,23);   // exp from float
//
//   ap_uint<BWOpr> exp = lhs_exp + rhs_exp;  // multiply of exp
//
//   // Underflow check with re-adjusting bias
//   if (exp > BIAS) { exp -= BIAS; }
//   else { exp = 0; }
//
//   // Output is float, thus exp can be remained as 8bit
//   rhs = (lhs_sign ^ rhs_sign, exp.range(7,0), rhs.range(22,0)); 
//
//   unsigned int result = rhs; // unavoidable type cast to native
//
//   return *(reinterpret_cast<float*>(&result));
    return lhs * rhs_native;
}

template <unsigned int BW>
__device__ float mul_fixed(fixedp<BW, __MAX_IW__> lhs, float rhs){
  return lhs * rhs;
}

__device__ float Number::operator*(const float rhs) const{
  switch (_type){
    case EXP:
      if (_bwTotal == 2) return mul_exp<2>(buf_exp2, rhs);
      else if (_bwTotal == 3) return mul_exp<3>(buf_exp3, rhs);
      else if (_bwTotal == 4) return mul_exp<4>(buf_exp4, rhs);
      else if (_bwTotal == 5) return mul_exp<5>(buf_exp5, rhs);
      else if (_bwTotal == 6) return mul_exp<6>(buf_exp6, rhs);
      else if (_bwTotal == 7) return mul_exp<7>(buf_exp7, rhs);
      else if (_bwTotal == 8) return mul_exp<8>(buf_exp8, rhs);
      else if (_bwTotal == 9) return mul_exp<9>(buf_exp9, rhs);
      else if (_bwTotal == 16) return mul_exp<16>(buf_exp, rhs);
      else {
        /* LOG(ERROR) << "Number::operator*(), Not a valid bitwidth " */
        /*            << __RED__ << _bwTotal << __END__; */
        /* exit(-1); */
      }
    case FIXED: 
      if (_bwTotal == 2) return mul_fixed<2>(buf_fixed2, rhs);
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
      else if (_bwTotal == 32) return mul_fixed<32>(buf_fixed32, rhs);
      else{
        // LOG(ERROR) << "Number::operator*(), Not a valid bitwidth "
        //            << __RED__ << _bwTotal << __END__;
        /* exit(-1); */
      }
    case FLOAT: return mul_float(buf_float, rhs);
    case HALF: return mul_half(buf_half, rhs);
    default: break;
//       LOG(ERROR) << "Number::operator*(), Invalid type for lhs";
      /* exit(-1); */
  }
}


__global__ void parallelConv(int fil_h, int fil_w, int fil_in,
                            int str_h, int str_v, int pad_h, int pad_w,
                            int bot_h, int bot_w,
                            int top_h, int top_w,
                            int bot_wst, int bot_hst, int bot_cst, int bot_nst,
                            int top_wst, int top_hst, int top_cst, int top_nst,
                            Dtype* top, const Dtype* bottom, const Number* f_gpu) {
  for (int w = 0; w < top_w; ++w) {
    Dtype sum = 0.0;
    extern __shared__ float cache[];
    for (int fh = 0; fh < fil_h; ++fh) {
      for (int fw = 0; fw < fil_w; ++fw) {
        int in_h = threadIdx.x * str_h - pad_h + fh;
        int in_w = w * str_v - pad_w + fw;
        if (in_w >= 0 && in_w < bot_w && in_h >= 0 && in_h < bot_h) {
          sum += *(f_gpu+fw + fh*fil_w + blockIdx.z*fil_w*fil_h + blockIdx.y*fil_in*fil_w*fil_h) *
              (*(bottom + in_w*bot_wst + in_h*bot_hst + blockIdx.z*bot_cst + blockIdx.x*bot_nst));
        }
      }
    }
    *(top + w * top_wst + threadIdx.x * top_hst + blockIdx.y * top_cst + blockIdx.x * top_nst) += sum;
  }
}

__global__ void parallelConv2(int fil_h, int fil_w, int fil_in,
                            int str_h, int str_v, int pad_h, int pad_w,
                            int bot_h, int bot_w,
                            int top_h, int top_w,
                            int bot_wst, int bot_hst, int bot_cst, int bot_nst,
                            int top_wst, int top_hst, int top_cst, int top_nst,
                            Dtype* top, const Dtype* bottom, const Number* f_gpu) {
    extern __shared__ float cache[];
    Dtype sum = 0.0;
    for (int fh = 0; fh < fil_h; ++fh) {
        for (int fw = 0; fw < fil_w; ++fw) {
            int in_h = threadIdx.x * str_h - pad_h + fh;
            int in_w = threadIdx.y * str_v - pad_w + fw;
            if (in_w >= 0 && in_w < bot_w && in_h >= 0 && in_h < bot_h) {
                sum += *(f_gpu + fw + fh * fil_w + blockIdx.z * fil_w * fil_h + blockIdx.y * fil_in * fil_w * fil_h) * (*(bottom + in_w * bot_wst + in_h * bot_hst + blockIdx.z * bot_cst + blockIdx.x * bot_nst));
            }
        }
    }
   *(top + threadIdx.y * top_wst + threadIdx.x * top_hst + blockIdx.y * top_cst + blockIdx.x * top_nst) += sum;
}

namespace util
{
    template < typename T > std::string to_string( const T& n )
    {
        std::ostringstream stm ;
        stm << n ;
        return stm.str() ;
    }
}

int idx = 0; // Counter to index layers and its configuration in a cfg file
void opendnnConvolutionForward (opendnnHandle_t handle,
  opendnnTensorDescriptor_t bottom_desc, const Dtype* bottom,
  opendnnFilterDescriptor_t filter_desc, const Dtype* filter,
  opendnnConvolutionDescriptor_t conv_desc,
  opendnnTensorDescriptor_t top_desc, Dtype* top) {
    int bot_n, bot_c, bot_h, bot_w, bot_nst, bot_cst, bot_hst, bot_wst;
    int top_n, top_c, top_h, top_w, top_nst, top_cst, top_hst, top_wst;
    int pad_h, pad_w, str_h, str_w, ups_x, ups_y;
    int fil_out, fil_in, fil_h, fil_w;
    int group, b_offset, t_offset, f_offset;
    opendnnGetTensor4dDescriptor (bottom_desc, &bot_n, &bot_c, &bot_h, &bot_w,
        &bot_nst, &bot_cst, &bot_hst, &bot_wst);
    opendnnGetTensor4dDescriptor (top_desc, &top_n, &top_c, &top_h, &top_w,
        &top_nst, &top_cst, &top_hst, &top_wst);
    opendnnGetFilter4dDescriptor (filter_desc, &fil_out, &fil_in, &fil_h, &fil_w);
    opendnnGetConvolution2dDescriptor (conv_desc, &pad_h, &pad_w, &str_h, &str_w, &ups_x, &ups_y);

    // How to solve total and name
    int total = fil_out * fil_in * fil_h * fil_w;

    Number* f_buf = new Number[total];
    for (int i = 0; i < total; i++) {
        f_buf[i].set(util::to_string(idx));
        f_buf[i] = (float)filter[i];
    }

    // Move to GPU
    Number* f_gpu;
    cudaMalloc(&f_gpu, sizeof(Number)*total);
    cudaMemcpy(f_gpu, f_buf, sizeof(Number)*total, cudaMemcpyHostToDevice);
    cudaMemset(top, 0, sizeof(top_wst*top_hst*top_cst*top_nst));

    if (top_h > 32) {
        dim3 dimBlock(top_h);
        dim3 dimGrid(top_n, top_c, bot_c);
        parallelConv<<<dimGrid, dimBlock, top_h>>>(fil_h, fil_w, fil_in,
                                    str_h, str_w, pad_h, pad_w,
                                    bot_h, bot_w,
                                    top_h, top_w,
                                    bot_wst, bot_hst, bot_cst, bot_nst,
                                    top_wst, top_hst, top_cst, top_nst,
                                    top, bottom, f_gpu);
    } else {
        dim3 dimBlock(top_h, top_w);
        dim3 dimGrid(top_n, top_c, bot_c);
        parallelConv2<<<dimGrid, dimBlock, top_h*top_w>>>(fil_h, fil_w, fil_in,
                                    str_h, str_w, pad_h, pad_w,
                                    bot_h, bot_w,
                                    top_h, top_w,
                                    bot_wst, bot_hst, bot_cst, bot_nst,
                                    top_wst, top_hst, top_cst, top_nst,
                                    top, bottom, f_gpu);
    }
    cudaFree(f_gpu);
    delete [] f_buf;

    DEBUG(bottom);
    DEBUG(top);
    idx++;
}


#endif //OPENDNN
