#ifdef __TEST__
#include <cstdlib>
#include "NumberADT.hpp"
#include "NumberADT_kernel.cuh"

using namespace std;


// This is just for testing
const int CNT = 20;
void top_accel(float input[CNT], Number weight[CNT], float fweight[CNT]){
    cerr << "EXP" << "   vs   " << "FLOAT" << endl;
    for (int i=1; i<CNT; i++)
       cerr << weight[i] * input[i] << "\t" << fweight[i] * input[i] << endl;
    for (int i=1; i<CNT; i++)
       cerr << weight[i] << "\t" << fweight[i] << endl;
}

int main() {
    float input[CNT];
    float fweight[CNT];
    Number *weight = new Number[CNT];

    for (int i=0; i<CNT; i++) {
        weight[i].init(EXP, 8, 0);
    }

    for (int i=1; i<CNT; i++){
        /* weight[i] = fweight[i] = (float)(0.0001245/i); */
        /* input[i] = (float)(0.03929/i); */
        weight[i] = fweight[i] = (float)(-2/powf(2,i));
        input[i] = (float)(4/powf(2,i));
    }

    top_accel(input, weight, fweight);
    delete [] weight;

    return 0;
}
#endif
