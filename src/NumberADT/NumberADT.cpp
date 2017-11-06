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
        std::cout << name << endl;
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
        return LOG2;
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
    log2quant temp(_bwTotal);
    temp = buf;
    return temp.getResult();
}

float Number::get_fixed() {
    fixedp temp(_bwTotal, __MAX_IW__);
    temp = buf;
    return (1 - temp.getSign()) * (float)temp.getData() / exp2f(temp.getTotal());
}

float Number::get_half() {
    sixteen<1> temp;
    temp = buf;
    return temp.getData();
}

// Overloading << for cout of Number
ostream& operator<<(ostream& os, Number& num) {
  switch (num.get_type()){
    case LOG2: os << num.get_exp(); break;
    case FIXED: os << num.get_fixed(); break;
    case FLOAT: os << num.get_float(); break;
    case HALF: os << num.get_half(); break;
    default:
      // LOG(ERROR) << "Number::operator<<(), Invalid type for print target";
      exit(-1);
  }
  return os;
}
