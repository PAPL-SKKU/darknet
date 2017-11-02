#ifndef NUMBER_H_
#define NUMBER_H_

#define __BW_OFF__ 0
#define __MAX_IW__ 0

#include <string>
#include <map>

#include "log2.hpp"
#include "fixed.hpp"
#include "half.hpp"

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

using namespace std;


enum DataType {
  EXP,
  FIXED,
  FLOAT,
  HALF
};

// Actual type for buffers (FLOAT, HALF, EXP, FIXED)
typedef float float_t;

const unsigned int BWOpr = 9 - 1 + 1;
// typedef ap_uint<16> exp_t;
typedef short exp_t;
// typedef ap_fixed<32, 16, AP_RND_CONV, AP_SAT>  fixed_t;

// Number type
class Number {
public:
  CUDA_HOSTDEV Number(){};
  CUDA_HOSTDEV Number(string name); // FLOAT
  CUDA_HOSTDEV Number(const float value, DataType type, int bwTotal, int bwInt=0); // FLOAT
  CUDA_HOSTDEV Number(DataType type, int bwTotal, int bwInt=0);
  CUDA_HOSTDEV void set(string name);
  CUDA_HOSTDEV void init(DataType type, int bwTotal, int bwInt=0);
  CUDA_HOSTDEV Number& operator*(Number& rhs) const;
  CUDA_HOSTDEV float operator*(float rhs);
  CUDA_HOSTDEV void operator=(const double rhs);
  CUDA_HOSTDEV void operator=(const float rhs);
  CUDA_HOSTDEV void operator=(const int rhs);
  CUDA_HOSTDEV void operator=(const Number& rhs);

  // Packs information into a single class (TypeInfo)
  class TypeInfo {
  public:
    TypeInfo(){}
    TypeInfo(DataType type, int bwTotal, int bwInt){
      _type = type;
      _bwTotal = bwTotal;
      _bwInt = bwInt;
    }
    friend class Number;
  private:
    DataType _type;
    int _bwTotal;
    int _bwInt;
  };

  // Configuration hash which holds <"conv1", TypeInfo(FLOAT, 32)> pair
  static map<string,TypeInfo> cfg;
  static void ParseConfigs(string filename);
  // thrust::pair<string,TypeInfo> cfg;
  // void ParseConfigs(string filename);

  // Getter
  DataType get_type(){return _type;}
  int get_bwTotal(){return _bwTotal;}
  int get_bwInt(){return _bwInt;}
  float get_exp();
  float get_fixed();
  float_t get_float(){return buf_float;}
  float get_half();
 
  bool is(DataType type){if (type == _type) return true; else return false;}

private:
  // Static buffer
  float_t buf_float;
  sixteen<1> buf_half;
  fixedp<32, __MAX_IW__> buf_fixed;
  fixedp<1, __MAX_IW__> buf_fixed1;
  fixedp<2, __MAX_IW__> buf_fixed2;
  fixedp<3, __MAX_IW__> buf_fixed3;
  fixedp<4, __MAX_IW__> buf_fixed4;
  fixedp<5, __MAX_IW__> buf_fixed5;
  fixedp<6, __MAX_IW__> buf_fixed6;
  fixedp<7, __MAX_IW__> buf_fixed7;
  fixedp<8, __MAX_IW__> buf_fixed8;
  fixedp<9, __MAX_IW__> buf_fixed9;
  fixedp<10, __MAX_IW__> buf_fixed10;
  fixedp<11, __MAX_IW__> buf_fixed11;
  fixedp<12, __MAX_IW__> buf_fixed12;
  fixedp<13, __MAX_IW__> buf_fixed13;
  fixedp<14, __MAX_IW__> buf_fixed14;
  fixedp<15, __MAX_IW__> buf_fixed15;
  fixedp<16, __MAX_IW__> buf_fixed16;
  fixedp<17, __MAX_IW__> buf_fixed17;
  fixedp<18, __MAX_IW__> buf_fixed18;
  fixedp<19, __MAX_IW__> buf_fixed19;
  fixedp<20, __MAX_IW__> buf_fixed20;
  fixedp<21, __MAX_IW__> buf_fixed21;
  fixedp<22, __MAX_IW__> buf_fixed22;
  fixedp<23, __MAX_IW__> buf_fixed23;
  fixedp<24, __MAX_IW__> buf_fixed24;
  fixedp<25, __MAX_IW__> buf_fixed25;
  fixedp<26, __MAX_IW__> buf_fixed26;
  fixedp<27, __MAX_IW__> buf_fixed27;
  fixedp<28, __MAX_IW__> buf_fixed28;
  fixedp<29, __MAX_IW__> buf_fixed29;
  fixedp<30, __MAX_IW__> buf_fixed30;
  fixedp<31, __MAX_IW__> buf_fixed31;
  fixedp<32, __MAX_IW__> buf_fixed32;
  log2quant<16> buf_exp;
  log2quant<1> buf_exp1;
  log2quant<2> buf_exp2;
  log2quant<3> buf_exp3;
  log2quant<4> buf_exp4;
  log2quant<5> buf_exp5;
  log2quant<6> buf_exp6;
  log2quant<7> buf_exp7;
  log2quant<8> buf_exp8;
  log2quant<9> buf_exp9;

  // Type info 
  DataType _type;
  short _bwTotal;
  short _bwInt;
};
// __attribute__ ((packed, aligned(32)));
ostream& operator<<(ostream& os, Number& num);
enum open_convolution_mode {
    CONVOLUTION,
    CROSS_CORRELATION
};


#endif // NUMBER_H_
