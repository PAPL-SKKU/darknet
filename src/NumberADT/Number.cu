#ifdef OPENDNN

#include <iostream>
#include <sstream>
#include <fstream>
#include <iterator>
#include <vector>
#include <string.h>
#include "debug.hpp"
#include "Number.hpp"

// static config variable
map<string,Number::TypeInfo> Number::cfg;

Number::Number() {
  _type = EXP;
  _bwTotal = 32;
  _bwInt = 0;
}

// Declaration w/o init, Number(<type>, <Total bitwidth>, <Integer bitwidth>)
// FLOAT / HALF / EXP / FIXED
Number::Number(DataType type, int bwTotal, int bwInt) {
  init(type, bwTotal, bwInt);
}

Number::Number(string name) {
  if (!cfg.count(name)){
    exit(-1);
  }
  TypeInfo config = cfg[name];
  if (config._type == FLOAT && config._bwTotal == 4) config._type == HALF;
  init(config._type, config._bwTotal, config._bwInt);
}

void Number::set(string name) {
    if (!cfg.count(name)){
        exit(-1);
    }
    TypeInfo config = cfg[name];
    if (config._type == FLOAT && config._bwTotal == 4) config._type == HALF;
    init(config._type, config._bwTotal, config._bwInt);
}

void Number::init(DataType type, int bwTotal, int bwInt){
  switch (type){
    case EXP  : buf_exp   = 0; break;
    case FIXED: buf_fixed = 0.f; break;
    case FLOAT: buf_float = 0; break;
    case HALF: buf_half = 0; break;
    default:
      exit(-1);
  }

  _type = type; 
  if(type == FIXED){
    _bwTotal = pow(2,bwTotal) + __BW_OFF__;
  }
  else{
    _bwTotal = pow(2,bwTotal);
  }
  _bwInt = 0;
}

// To initialize a single element, Number(<value>, <type>, <Total bitwidth>)
// FLOAT TYPE
Number::Number(const float value, DataType type, int bwTotal, int bwInt) {
  switch (type){
    case EXP  : buf_exp   = value; break;
    case FIXED: buf_fixed = value; break;
    case FLOAT: buf_float = value; break;
    case HALF: buf_half = value; break;
    default:
      exit(-1);
  }
 
  _type = type; 
  if(type == FIXED){
    _bwTotal = bwTotal + __BW_OFF__;
  }
  else{
    _bwTotal = bwTotal;
  }
  _bwInt = 0;
}

// template<unsigned int BW>
// log2quant<BW> convertToExp(float data){
//   const unsigned int BIAS = pow(2,BW-2) - 1;
//   const unsigned int EXP_MAX = pow(2,BW-1) - 1;
//
//   log2quant<32> temp = *reinterpret_cast<unsigned int*>(&data);
//   log2quant<BW-1> exp;
//
//   // Underflow check 
//   log2quant<32> check = temp.range(30,23);
//   if (check > (127-BIAS)){
//     check -= (127-BIAS);
//   } else {
//     // Underflow occurs
//     check = 0;
//     // LOG(WARNING) << "Number::operator=(), underflow occurs!";
//     // exit(-1);
//   }
//
//   // Overflow check
//   if (check > EXP_MAX){
// //     LOG(ERROR) << "Number::operator=(), overflow occurs!";
//     exit(-1);
//   } else {
//     exp = check;
//   }
//
//   log2quant<BW> result = (temp[31], exp);
//   return result;
// }
// template<unsigned int BW>
// ap_fixed<BW, __MAX_IW__, AP_RND_CONV, AP_SAT> convertToFixed(float data){
  // return (fixedp<32, __MAX_IW__>)data;
// }

void Number::operator=(float rhs){
  switch (_type){
    case EXP:
      if (_bwTotal == 2) buf_exp2 = rhs;
      else if (_bwTotal == 3) buf_exp3 = rhs;
      else if (_bwTotal == 4) buf_exp4 = rhs;
      else if (_bwTotal == 5) buf_exp5 = rhs;
      else if (_bwTotal == 6) buf_exp6 = rhs;
      else if (_bwTotal == 7) buf_exp7 = rhs;
      else if (_bwTotal == 8) buf_exp8 = rhs;
      else if (_bwTotal == 9) buf_exp9 = rhs;
      else if (_bwTotal == 16) buf_exp = rhs;
      else {
        LOG(ERROR) << "Number::operator=(), Not a valid EXP bitwidth "
                   << __RED__ << _bwTotal << __END__;
        exit(-1);
      }
    break;
    case FIXED: 
      if (_bwTotal == 2) buf_fixed2 = rhs; 
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
      else if (_bwTotal == 32) buf_fixed32 = rhs; 
      else{
          LOG(ERROR) << "Number::operator=(), Not a valid FIXED bitwidth "
                     << __RED__ << _bwTotal << __END__;
          exit(-1);
      }
      break;
    case FLOAT: buf_float = (float_t) rhs; break;
    case HALF: buf_half = rhs; break;
    default:
      LOG(ERROR) << "Number::operator*(), Invalid type for lhs";
      exit(-1);
  }
}

// TODO: Wrong type conversion, currently just forced, value must be treated
void Number::operator=(const int rhs){
  switch (_type){
    case EXP  : buf_exp   = (float)rhs; break;
    case FIXED: buf_fixed = (float)rhs; break;
    case FLOAT: buf_float = (float_t) rhs; break;
    case HALF: buf_half = (float)rhs; break;
    default:
      exit(-1);
  }
}


// TODO: Is it necessaray to support all those 9 combinations?
Number& Number::operator=(const Number& rhs){
  switch (_type){
    case EXP  : buf_exp   = rhs.buf_exp; break;
    case FIXED: buf_fixed = rhs.buf_fixed; break;
    case FLOAT: buf_float = rhs.buf_float; break;
    case HALF:  buf_half = rhs.buf_half; break;
    default:
      LOG(ERROR) << "Number::operator*(), Invalid type for lhs";
      exit(-1);
  }
}

namespace util {
// Utility functions mostly for parsing the configuration file
template<typename output>
void split(const string &s, char delim, output result) {
    stringstream ss;
    ss.str(s);
    string item;
    while (getline(ss, item, delim)) {
        *(result++) = item;
    }
}

// Split string into vectors with a delimiter
vector<string> split(const string &s, char delim) {
    vector<string> elems;
    split(s, delim, back_inserter(elems));
    return elems;
}

// Convert string type "FLOAT" to DataType FLOAT (enum)
DataType atot(string type){
    if(type == "FLOAT"){
        return FLOAT;
    } else if(type == "HALF"){
        return HALF; 
    } else if(type == "FIXED"){
        return FIXED;
    } else if(type == "EXP"){
        return EXP;
    } else {
        LOG(ERROR) << "Number::ParseConfigs(), util::atot ERROR";
        exit(-1);
    }
}

};

// Parse configuration files
// e.g. conv1 FLOAT 32 --> {"conv1":TypeInfo(FLOAT, 32)}
void Number::ParseConfigs(string filename){
  ifstream cfgFile("config.txt");
  string line;
  while (getline(cfgFile, line)) {
    vector<string> tmp = util::split(line, '\t');

    // Key:   "conv1"
    // Value: FLOAT, 32
    // Value: FIXED, 32, 16
    switch(tmp.size()){
      case 3:
        cfg.insert(pair<string, TypeInfo>(
           tmp[0],  // key
           TypeInfo(util::atot(tmp[1]), atoi(tmp[2].c_str()), 0 ))  // value
        );
        break;
      case 4:
        cfg.insert(pair<string, TypeInfo>(
           tmp[0],  // key
           TypeInfo(util::atot(tmp[1]), atoi(tmp[2].c_str()), atoi(tmp[3].c_str()) ))  // value
        );
        break;
      default:
        LOG(ERROR) << "Number::ParseConfigs, Invalid configuration format";
        break;
    } 
  }

  cfgFile.close();
}

// Getter for arbitrary bitwidth of exp
float Number::get_exp(){
  if (_bwTotal == 2) return buf_exp2.getResult();
  else if (_bwTotal == 3) return buf_exp3.getResult();
  else if (_bwTotal == 4) return buf_exp4.getResult();
  else if (_bwTotal == 5) return buf_exp5.getResult();
  else if (_bwTotal == 6) return buf_exp6.getResult();
  else if (_bwTotal == 7) return buf_exp7.getResult();
  else if (_bwTotal == 8) return buf_exp8.getResult();
  else if (_bwTotal == 9) return buf_exp9.getResult();
  else if (_bwTotal == 16) return buf_exp.getResult();
  else {
    LOG(ERROR) << "Number::get_exp(), Not a valid bitwidth "
               << __RED__ << _bwTotal << __END__;
    exit(-1);
  }
}

float Number::get_fixed(){
  // if (_bwTotal == 2) return fixedp<32, __MAX_IW__>(buf_fixed2);
  // else if (_bwTotal == 3) return fixedp<32, __MAX_IW__>(buf_fixed3);
  // else if (_bwTotal == 4) return fixedp<32, __MAX_IW__>(buf_fixed4);
  // else if (_bwTotal == 5) return fixedp<32, __MAX_IW__>(buf_fixed5);
  // else if (_bwTotal == 6) return fixedp<32, __MAX_IW__>(buf_fixed6);
  // else if (_bwTotal == 7) return fixedp<32, __MAX_IW__>(buf_fixed7);
  // else if (_bwTotal == 8) return fixedp<32, __MAX_IW__>(buf_fixed8);
  // else if (_bwTotal == 9) return fixedp<32, __MAX_IW__>(buf_fixed9);
  // else if (_bwTotal == 10) return fixedp<32, __MAX_IW__>(buf_fixed10);
  // else if (_bwTotal == 11) return fixedp<32, __MAX_IW__>(buf_fixed11);
  // else if (_bwTotal == 12) return fixedp<32, __MAX_IW__>(buf_fixed12);
  // else if (_bwTotal == 13) return fixedp<32, __MAX_IW__>(buf_fixed13);
  // else if (_bwTotal == 14) return fixedp<32, __MAX_IW__>(buf_fixed14);
  // else if (_bwTotal == 15) return fixedp<32, __MAX_IW__>(buf_fixed15);
  // else if (_bwTotal == 16) return fixedp<32, __MAX_IW__>(buf_fixed16);
  // else if (_bwTotal == 32) return fixedp<32, __MAX_IW__>(buf_fixed32);
  // else{
  //   // LOG(ERROR) << "Number::get_fixed(), Not a valid bitwidth"
  //   //            << __RED__ << _bwTotal << __END__;
  //   exit(-1);
  // }
  if (_bwTotal == 2) return (1 - buf_fixed2.getSign())*(float)buf_fixed2.getData() / exp2f(buf_fixed2.getTotal());
  else if (_bwTotal == 3) return (1-buf_fixed3.getSign())*(float)buf_fixed3.getData() / exp2f(buf_fixed3.getTotal());
  else if (_bwTotal == 4) return (1-buf_fixed4.getSign())*(float)buf_fixed4.getData() / exp2f(buf_fixed4.getTotal());
  else if (_bwTotal == 5) return (1-buf_fixed5.getSign())*(float)buf_fixed5.getData() / exp2f(buf_fixed5.getTotal());
  else if (_bwTotal == 6) return (1-buf_fixed6.getSign())*(float)buf_fixed6.getData() / exp2f(buf_fixed6.getTotal());
  else if (_bwTotal == 7) return (1-buf_fixed7.getSign())*(float)buf_fixed7.getData() / exp2f(buf_fixed7.getTotal());
  else if (_bwTotal == 8) return (1-buf_fixed8.getSign())*(float)buf_fixed8.getData() / exp2f(buf_fixed8.getTotal());
  else if (_bwTotal == 9) return (1-buf_fixed9.getSign())*(float)buf_fixed9.getData() / exp2f(buf_fixed9.getTotal());
  else if (_bwTotal == 10) return (1-buf_fixed10.getSign())*(float)buf_fixed10.getData() / exp2f(buf_fixed10.getTotal());
  else if (_bwTotal == 11) return (1-buf_fixed11.getSign())*(float)buf_fixed11.getData() / exp2f(buf_fixed11.getTotal());
  else if (_bwTotal == 12) return (1-buf_fixed12.getSign())*(float)buf_fixed12.getData() / exp2f(buf_fixed12.getTotal());
  else if (_bwTotal == 13) return (1-buf_fixed13.getSign())*(float)buf_fixed13.getData() / exp2f(buf_fixed13.getTotal());
  else if (_bwTotal == 14) return (1-buf_fixed14.getSign())*(float)buf_fixed14.getData() / exp2f(buf_fixed14.getTotal());
  else if (_bwTotal == 15) return (1-buf_fixed15.getSign())*(float)buf_fixed15.getData() / exp2f(buf_fixed15.getTotal());
  else if (_bwTotal == 16) return (1-buf_fixed16.getSign())*(float)buf_fixed16.getData() / exp2f(buf_fixed16.getTotal());
  else if (_bwTotal == 32) return (1-buf_fixed32.getSign())*(float)buf_fixed32.getData() / exp2f(buf_fixed32.getTotal());
  else {
    LOG(ERROR) << "Number::get_fixed(), Not a valid bitwidth"
               << __RED__ << _bwTotal << __END__;
    exit(-1);
  }
}

float Number::get_half() {
    return buf_half.getData();
}

// Overloading << for cout of Number
ostream& operator<<(ostream& os, Number& num) {
  switch (num.get_type()){
    case EXP  : os << num.get_exp(); break;
    case FIXED: os << num.get_fixed(); break;
    case FLOAT: os << num.get_float(); break;
    case HALF: os << num.get_half(); break;
    default:
      LOG(ERROR) << "Number::operator<<(), Invalid type for print target";
      exit(-1);
  }
  return os;
}

#endif
