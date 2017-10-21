#ifndef NUMBER_H
#define NUMBER_H

#define __BW_OFF__ 0
#define __MAX_IW__ 0

#include <string>
#include <map>
#include <iostream>

#include "NumberADT/half.hpp"
#include "NumberADT/log2.hpp"
#include "NumberADT/fixed.hpp"

// #ifdef __CUDACC__
// #define CUDA_HOSTDEV __device__
// #else
// #define CUDA_HOSTDEV
// #endif

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
  Number();
  Number(string name); // FLOAT
  Number(const float value, DataType type, int bwTotal, int bwInt=0); // FLOAT
  Number(DataType type, int bwTotal, int bwInt=0);
  void set(string name);
  void init(DataType type, int bwTotal, int bwInt=0);
  // CUDA_HOSTDEV Number& operator*(Number& rhs) const;
  __device__ float operator*(const float rhs) const;
  void operator=(const double rhs);
  void operator=(const float rhs);
  void operator=(const int rhs);
  Number& operator=(const Number& rhs);

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
  // ap_fixed<2, __MAX_IW__, AP_RND_CONV, AP_SAT> buf_fixed2;
  // ap_fixed<3, __MAX_IW__, AP_RND_CONV, AP_SAT> buf_fixed3;
  // ap_fixed<4, __MAX_IW__, AP_RND_CONV, AP_SAT> buf_fixed4;
  // ap_fixed<5, __MAX_IW__, AP_RND_CONV, AP_SAT> buf_fixed5;
  // ap_fixed<6, __MAX_IW__, AP_RND_CONV, AP_SAT> buf_fixed6;
  // ap_fixed<7, __MAX_IW__, AP_RND_CONV, AP_SAT> buf_fixed7;
  // ap_fixed<8, __MAX_IW__, AP_RND_CONV, AP_SAT> buf_fixed8;
  // ap_fixed<9, __MAX_IW__, AP_RND_CONV, AP_SAT> buf_fixed9;
  // ap_fixed<10, __MAX_IW__, AP_RND_CONV, AP_SAT> buf_fixed10;
  // ap_fixed<11, __MAX_IW__, AP_RND_CONV, AP_SAT> buf_fixed11;
  // ap_fixed<12, __MAX_IW__, AP_RND_CONV, AP_SAT> buf_fixed12;
  // ap_fixed<13, __MAX_IW__, AP_RND_CONV, AP_SAT> buf_fixed13;
  // ap_fixed<14, __MAX_IW__, AP_RND_CONV, AP_SAT> buf_fixed14;
  // ap_fixed<15, __MAX_IW__, AP_RND_CONV, AP_SAT> buf_fixed15;
  // ap_fixed<16, __MAX_IW__, AP_RND_CONV, AP_SAT> buf_fixed16;
  // ap_fixed<32, __MAX_IW__, AP_RND_CONV, AP_SAT> buf_fixed32;
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
  fixedp<32, __MAX_IW__> buf_fixed32;
  log2quant<16> buf_exp;
  // ap_uint<2> buf_exp2;
  // ap_uint<3> buf_exp3;
  // ap_uint<4> buf_exp4;
  // ap_uint<5> buf_exp5;
  // ap_uint<6> buf_exp6;
  // ap_uint<7> buf_exp7;
  // ap_uint<8> buf_exp8;
  // ap_uint<9> buf_exp9;
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

#endif
