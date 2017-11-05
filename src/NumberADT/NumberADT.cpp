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
  buf = 0;
  _bwTotal = bwTotal;
  _type = type; 
  if(_type == FIXED){
    _bwTotal = bwTotal +__BW_OFF__;
  }
  _bwInt = bwInt;
}

// To initialize a single element, Number(<value>, <type>, <Total bitwidth>)
// FLOAT TYPE
Number::Number(const float value, DataType type, int bwTotal, int bwInt) {
    buf = value;
    _bwTotal = bwTotal;
    _type = type; 
    if(_type == FIXED){
        _bwTotal += __BW_OFF__;
    _bwInt = bwInt;
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
    if (_bwTotal == 2) {
        log2quant<2> temp;
        temp = buf;
        return temp.getResult();
    } else if (_bwTotal == 3) {
        log2quant<3> temp;
        temp = buf;
        return temp.getResult();
    } else if (_bwTotal == 4) {
        log2quant<4> temp;
        temp = buf;
        return temp.getResult();
    } else if (_bwTotal == 5) {
        log2quant<5> temp;
        temp = buf;
        return temp.getResult();
    } else if (_bwTotal == 6) {
        log2quant<6> temp;
        temp = buf;
        return temp.getResult();
    } else if (_bwTotal == 7) {
        log2quant<7> temp;
        temp = buf;
        return temp.getResult();
    } else if (_bwTotal == 8) {
        log2quant<8> temp;
        temp = buf;
        return temp.getResult();
    } else if (_bwTotal == 9) {
        log2quant<9> temp;
        temp = buf;
        return temp.getResult();
    } else if (_bwTotal == 16) {
        log2quant<16> temp;
        temp = buf;
        return temp.getResult();
    }
}

float Number::get_fixed() {
    if (_bwTotal == 1) {
        fixedp<1,__MAX_IW__> temp;
        temp = buf;
        return (1 - 2*temp.getSign()) * (float)temp.getData() / exp2f(temp.getTotal());
    } else if (_bwTotal == 2) {
        fixedp<2,__MAX_IW__> temp;
        temp = buf;
        return (1 - 2*temp.getSign()) * (float)temp.getData() / exp2f(temp.getTotal());
    } else if (_bwTotal == 3) {
        fixedp<3,__MAX_IW__> temp;
        temp = buf;
        return (1 - 2*temp.getSign()) * (float)temp.getData() / exp2f(temp.getTotal());
    } else if (_bwTotal == 4) {
        fixedp<4,__MAX_IW__> temp;
        temp = buf;
        return (1 - 2*temp.getSign()) * (float)temp.getData() / exp2f(temp.getTotal());
    } else if (_bwTotal == 5) {
        fixedp<5,__MAX_IW__> temp;
        temp = buf;
        return (1 - 2*temp.getSign()) * (float)temp.getData() / exp2f(temp.getTotal());
    } else if (_bwTotal == 6) {
        fixedp<6,__MAX_IW__> temp;
        temp = buf;
        return (1 - 2*temp.getSign()) * (float)temp.getData() / exp2f(temp.getTotal());
    } else if (_bwTotal == 7) {
        fixedp<7,__MAX_IW__> temp;
        temp = buf;
        return (1 - 2*temp.getSign()) * (float)temp.getData() / exp2f(temp.getTotal());
    } else if (_bwTotal == 8) {
        fixedp<8,__MAX_IW__> temp;
        temp = buf;
        return (1 - 2*temp.getSign()) * (float)temp.getData() / exp2f(temp.getTotal());
    } else if (_bwTotal == 9) {
        fixedp<9,__MAX_IW__> temp;
        temp = buf;
        return (1 - 2*temp.getSign()) * (float)temp.getData() / exp2f(temp.getTotal());
    } else if (_bwTotal == 10) {
        fixedp<10,__MAX_IW__> temp;
        temp = buf;
        return (1 - 2*temp.getSign()) * (float)temp.getData() / exp2f(temp.getTotal());
    } else if (_bwTotal == 11) {
        fixedp<11,__MAX_IW__> temp;
        temp = buf;
        return (1 - 2*temp.getSign()) * (float)temp.getData() / exp2f(temp.getTotal());
    } else if (_bwTotal == 12) {
        fixedp<12,__MAX_IW__> temp;
        temp = buf;
        return (1 - 2*temp.getSign()) * (float)temp.getData() / exp2f(temp.getTotal());
    } else if (_bwTotal == 13) {
        fixedp<13,__MAX_IW__> temp;
        temp = buf;
        return (1 - 2*temp.getSign()) * (float)temp.getData() / exp2f(temp.getTotal());
    } else if (_bwTotal == 14) {
        fixedp<14,__MAX_IW__> temp;
        temp = buf;
        return (1 - 2*temp.getSign()) * (float)temp.getData() / exp2f(temp.getTotal());
    } else if (_bwTotal == 15) {
        fixedp<15,__MAX_IW__> temp;
        temp = buf;
        return (1 - 2*temp.getSign()) * (float)temp.getData() / exp2f(temp.getTotal());
    } else if (_bwTotal == 16) {
        fixedp<16,__MAX_IW__> temp;
        temp = buf;
        return (1 - 2*temp.getSign()) * (float)temp.getData() / exp2f(temp.getTotal());
    } else if (_bwTotal == 17) {
        fixedp<17,__MAX_IW__> temp;
        temp = buf;
        return (1 - 2*temp.getSign()) * (float)temp.getData() / exp2f(temp.getTotal());
    } else if (_bwTotal == 18) {
        fixedp<18,__MAX_IW__> temp;
        temp = buf;
        return (1 - 2*temp.getSign()) * (float)temp.getData() / exp2f(temp.getTotal());
    } else if (_bwTotal == 19) {
        fixedp<19,__MAX_IW__> temp;
        temp = buf;
        return (1 - 2*temp.getSign()) * (float)temp.getData() / exp2f(temp.getTotal());
    } else if (_bwTotal == 20) {
        fixedp<20,__MAX_IW__> temp;
        temp = buf;
        return (1 - 2*temp.getSign()) * (float)temp.getData() / exp2f(temp.getTotal());
    } else if (_bwTotal == 21) {
        fixedp<21,__MAX_IW__> temp;
        temp = buf;
        return (1 - 2*temp.getSign()) * (float)temp.getData() / exp2f(temp.getTotal());
    } else if (_bwTotal == 22) {
        fixedp<22,__MAX_IW__> temp;
        temp = buf;
        return (1 - 2*temp.getSign()) * (float)temp.getData() / exp2f(temp.getTotal());
    } else if (_bwTotal == 23) {
        fixedp<23,__MAX_IW__> temp;
        temp = buf;
        return (1 - 2*temp.getSign()) * (float)temp.getData() / exp2f(temp.getTotal());
    } else if (_bwTotal == 24) {
        fixedp<24,__MAX_IW__> temp;
        temp = buf;
        return (1 - 2*temp.getSign()) * (float)temp.getData() / exp2f(temp.getTotal());
    } else if (_bwTotal == 25) {
        fixedp<25,__MAX_IW__> temp;
        temp = buf;
        return (1 - 2*temp.getSign()) * (float)temp.getData() / exp2f(temp.getTotal());
    } else if (_bwTotal == 26) {
        fixedp<26,__MAX_IW__> temp;
        temp = buf;
        return (1 - 2*temp.getSign()) * (float)temp.getData() / exp2f(temp.getTotal());
    } else if (_bwTotal == 27) {
        fixedp<27,__MAX_IW__> temp;
        temp = buf;
        return (1 - 2*temp.getSign()) * (float)temp.getData() / exp2f(temp.getTotal());
    } else if (_bwTotal == 28) {
        fixedp<28,__MAX_IW__> temp;
        temp = buf;
        return (1 - 2*temp.getSign()) * (float)temp.getData() / exp2f(temp.getTotal());
    } else if (_bwTotal == 29) {
        fixedp<29,__MAX_IW__> temp;
        temp = buf;
        return (1 - 2*temp.getSign()) * (float)temp.getData() / exp2f(temp.getTotal());
    } else if (_bwTotal == 30) {
        fixedp<30,__MAX_IW__> temp;
        temp = buf;
        return (1 - 2*temp.getSign()) * (float)temp.getData() / exp2f(temp.getTotal());
    } else if (_bwTotal == 31) {
        fixedp<31,__MAX_IW__> temp;
        temp = buf;
        return (1 - 2*temp.getSign()) * (float)temp.getData() / exp2f(temp.getTotal());
    } else {
        fixedp<32,__MAX_IW__> temp;
        temp = buf;
        return (1 - 2*temp.getSign()) * (float)temp.getData() / exp2f(temp.getTotal());
    }
}

float Number::get_half() {
    sixteen<1> temp;
    temp = buf;
    return temp.getData();
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
