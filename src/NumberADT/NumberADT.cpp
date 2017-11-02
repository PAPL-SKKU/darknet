#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <stdlib.h>

#include "debug.hpp"
#include "NumberADT.hpp"

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
  init(config._type, config._bwTotal, config._bwInt);
}

void Number::set(string name) {
    if (!cfg.count(name)){
        exit(-1);
    }
    TypeInfo config = cfg[name];
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
    _bwTotal = bwTotal + __BW_OFF__;
  }
  else{
    _bwTotal = bwTotal;
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
        cerr << "ERROR" << endl;
        exit(-1);
    }
}

};

// static config variable
map<string,Number::TypeInfo> Number::cfg;

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
    // LOG(ERROR) << "Number::get_exp(), Not a valid bitwidth "
    //            << __RED__ << _bwTotal << __END__;
    exit(-1);
  }
}

float Number::get_fixed(){
  if (_bwTotal == 1) return (1 - buf_fixed1.getSign())*(float)buf_fixed1.getData() / exp2f(buf_fixed1.getTotal());
  else if (_bwTotal == 2) return (1 - buf_fixed2.getSign())*(float)buf_fixed2.getData() / exp2f(buf_fixed2.getTotal());
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
  else if (_bwTotal == 17) return (1-buf_fixed17.getSign())*(float)buf_fixed17.getData() / exp2f(buf_fixed17.getTotal());
  else if (_bwTotal == 18) return (1-buf_fixed18.getSign())*(float)buf_fixed18.getData() / exp2f(buf_fixed18.getTotal());
  else if (_bwTotal == 19) return (1-buf_fixed19.getSign())*(float)buf_fixed19.getData() / exp2f(buf_fixed19.getTotal());
  else if (_bwTotal == 20) return (1-buf_fixed20.getSign())*(float)buf_fixed20.getData() / exp2f(buf_fixed20.getTotal());
  else if (_bwTotal == 21) return (1-buf_fixed21.getSign())*(float)buf_fixed21.getData() / exp2f(buf_fixed21.getTotal());
  else if (_bwTotal == 22) return (1-buf_fixed22.getSign())*(float)buf_fixed22.getData() / exp2f(buf_fixed22.getTotal());
  else if (_bwTotal == 23) return (1-buf_fixed23.getSign())*(float)buf_fixed23.getData() / exp2f(buf_fixed23.getTotal());
  else if (_bwTotal == 24) return (1-buf_fixed24.getSign())*(float)buf_fixed24.getData() / exp2f(buf_fixed24.getTotal());
  else if (_bwTotal == 25) return (1-buf_fixed25.getSign())*(float)buf_fixed25.getData() / exp2f(buf_fixed25.getTotal());
  else if (_bwTotal == 26) return (1-buf_fixed26.getSign())*(float)buf_fixed26.getData() / exp2f(buf_fixed26.getTotal());
  else if (_bwTotal == 27) return (1-buf_fixed27.getSign())*(float)buf_fixed27.getData() / exp2f(buf_fixed27.getTotal());
  else if (_bwTotal == 28) return (1-buf_fixed28.getSign())*(float)buf_fixed28.getData() / exp2f(buf_fixed28.getTotal());
  else if (_bwTotal == 29) return (1-buf_fixed29.getSign())*(float)buf_fixed29.getData() / exp2f(buf_fixed29.getTotal());
  else if (_bwTotal == 30) return (1-buf_fixed30.getSign())*(float)buf_fixed30.getData() / exp2f(buf_fixed30.getTotal());
  else if (_bwTotal == 31) return (1-buf_fixed31.getSign())*(float)buf_fixed31.getData() / exp2f(buf_fixed31.getTotal());
  else if (_bwTotal == 32) return (1-buf_fixed32.getSign())*(float)buf_fixed32.getData() / exp2f(buf_fixed32.getTotal());
  else {
    // LOG(ERROR) << "Number::get_fixed(), Not a valid bitwidth"
    //            << __RED__ << _bwTotal << __END__;
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
      // LOG(ERROR) << "Number::operator<<(), Invalid type for print target";
      exit(-1);
  }
  return os;
}
