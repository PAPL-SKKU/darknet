#ifdef __TEST__
#include <cstdlib>
#include "NumberADT.hpp"
#include "NumberADT_kernel.cuh"

using namespace std;


// This is just for testing
const int CNT = 20;
void top_accel(float input[CNT], Number weight[CNT], float fweight[CNT]){
    cerr << "FIXED" << "   vs   " << "FLOAT" << endl;
    for (int i=1; i<CNT; i++)
       cerr << weight[i] * input[i] << "\t" << fweight[i] * input[i] << endl;
    for (int i=1; i<CNT; i++)
       cerr << weight[i].get_float() << "\t" << fweight[i] << endl;
}

int main() {
    float input[CNT];
    float fweight[CNT];
    Number *weight = new Number[CNT];

    for (int i=0; i<CNT; i++) {
        weight[i].init(FIXED, 24, 0);
    }

    for (int i=1; i<CNT; i++){
        /* weight[i] = fweight[i] = (float)(0.0001245/i); */
        /* input[i] = (float)(0.03929/i); */
        weight[i] = fweight[i] = (float)(-3/powf(2,i));
        input[i] = (float)(6/powf(2,i));
    }

    top_accel(input, weight, fweight);
    delete [] weight;

    return 0;
}
#endif