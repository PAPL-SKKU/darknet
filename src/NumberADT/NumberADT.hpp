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
typedef short exp_t;

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

  // Getter
  DataType get_type(){return _type;}
  int get_bwTotal(){return _bwTotal;}
  int get_bwInt(){return _bwInt;}
  float get_exp();
  float get_fixed();
  float get_float(){return *reinterpret_cast<float*>(&buf);}
  float get_half();
 
  bool is(DataType type){if (type == _type) true; else false;}

private:
  float buf;

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
