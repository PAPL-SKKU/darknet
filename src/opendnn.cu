#include <cmath>
#include <iostream>
#include <iomanip>
#include <limits>

#include "NumberADT/NumberADT.hpp"

extern "C"{
#include "opendnn.h"
}
#include "opendnn_kernel.cuh"

#include <string>
#include <sstream>
#include <fstream>
#include <iterator>
#include <vector>

using namespace std;


void opendnnCreate (opendnnHandle_t* handle) {
    Number::ParseConfigs("config.txt");
    *handle = new opendnnContext;
}

void opendnnCreateTensorDescriptor (opendnnTensorDescriptor_t* tensor_desc) {
    *tensor_desc = new opendnnTensorStruct;
}

void opendnnSetTensor4dDescriptor (opendnnTensorDescriptor_t tens, int n, int c, int h, int w) {
    int w_str = 1;
    int h_str = w * w_str;
    int c_str = h * h_str;
    int n_str = c * c_str;
    opendnnSetTensor4dDescriptorEx (tens, n, c, h, w, n_str, c_str, h_str, w_str);
}

void opendnnSetTensor4dDescriptorEx (opendnnTensorDescriptor_t tens, int n, int c, int h, int w,
  int nStride, int cStride, int hStride, int wStride) {
    tens->number_ = n;
    tens->channel_ = c;
    tens->height_ = h;
    tens->width_ = w;
    tens->stride_n = nStride;
    tens->stride_c = cStride;
    tens->stride_h = hStride;
    tens->stride_w = wStride;
}

void opendnnGetTensor4dDescriptor (opendnnTensorDescriptor_t tens, int* n, int* c, int* h, int* w,
  int* nStride, int* cStride, int* hStride, int* wStride) {
    *n = tens->number_;
    *c = tens->channel_;
    *h = tens->height_;
    *w = tens->width_;
    *nStride = tens->stride_n;
    *cStride = tens->stride_c;
    *hStride = tens->stride_h;
    *wStride = tens->stride_w;
}


// Filter
void opendnnCreateFilterDescriptor (opendnnFilterDescriptor_t* filter) {
    *filter = new opendnnFilterStruct;
}

void opendnnSetFilter4dDescriptor (opendnnFilterDescriptor_t filter, int out, int in, int h, int w) {
    filter->output_ = out;
    filter->input_ = in;
    filter->height_ = h;
    filter->width_ = w;
}

void opendnnGetFilter4dDescriptor (opendnnFilterDescriptor_t filter, int* out, int* in, int* h, int* w) {
    *out = filter->output_;
    *in = filter->input_;
    *h = filter->height_;
    *w = filter->width_;
}


// Convolution
void opendnnCreateConvolutionDescriptor (opendnnConvolutionDescriptor_t* conv_desc) {
    *conv_desc = new opendnnConvolutionStruct;
    (*conv_desc)->group = 1;
}

void opendnnSetConvolution2dDescriptor (opendnnConvolutionDescriptor_t conv_desc, int ph, int pw, int sh, int sw, int ux, int uy) {
    conv_desc->pad_h = ph;
    conv_desc->pad_w = pw;
    conv_desc->vertical_stride = sw;
    conv_desc->horizon_stride = sh;
    conv_desc->upscale_x = ux;
    conv_desc->upscale_y = uy;
}

void opendnnGetConvolution2dDescriptor (opendnnConvolutionDescriptor_t conv_desc,
  int* ph, int* pw, int* sh, int* sw, int* ux, int* uy) {
    *ph = conv_desc->pad_h;
    *pw = conv_desc->pad_w;
    *sw = conv_desc->vertical_stride;
    *sh = conv_desc->horizon_stride;
    *ux = conv_desc->upscale_x;
    *uy = conv_desc->upscale_y;
}

void opendnnSetConvolutionGroupCount(opendnnConvolutionDescriptor_t conv_desc, int group) {
    conv_desc->group = group;
}

void opendnnGetConvolutionGroupCount(opendnnConvolutionDescriptor_t conv_desc, int* group) {
    *group = conv_desc->group;
}


void im2col_gpu(const float* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    float* data_col) {
  // We are going to launch channels * height_col * width_col kernels, each
  // kernel responsible for copying a single-channel grid.
  int height_col = (height + 2 * pad_h -
      (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  int width_col = (width + 2 * pad_w -
      (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  int num_kernels = channels * height_col * width_col;
  
  const int MAX_THREADS=1024; // Titan X/Xp's maximum available threads
  im2col_gpu_kernel<<<(num_kernels+MAX_THREADS-1)/MAX_THREADS, MAX_THREADS>>>(
      num_kernels, data_im, height, width, kernel_h, kernel_w, pad_h,
      pad_w, stride_h, stride_w, dilation_h, dilation_w, height_col,
      width_col, data_col);
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

int idx = 0;
void opendnnConvolutionForward (opendnnHandle_t handle,
  opendnnTensorDescriptor_t bottom_desc, float* bottom,
  opendnnFilterDescriptor_t filter_desc, float* filter,
  opendnnConvolutionDescriptor_t conv_desc,
  opendnnTensorDescriptor_t top_desc, float* top) {
    int bot_n, bot_c, bot_h, bot_w, bot_nst, bot_cst, bot_hst, bot_wst;
    int top_n, top_c, top_h, top_w, top_nst, top_cst, top_hst, top_wst;
    int pad_h, pad_w, str_h, str_w, ups_x, ups_y;
    int fil_out, fil_in, fil_h, fil_w;
    int group;
    opendnnGetTensor4dDescriptor (bottom_desc, &bot_n, &bot_c, &bot_h, &bot_w,
        &bot_nst, &bot_cst, &bot_hst, &bot_wst);
    opendnnGetTensor4dDescriptor (top_desc, &top_n, &top_c, &top_h, &top_w,
        &top_nst, &top_cst, &top_hst, &top_wst);
    opendnnGetFilter4dDescriptor (filter_desc, &fil_out, &fil_in, &fil_h, &fil_w);
    opendnnGetConvolution2dDescriptor (conv_desc, &pad_h, &pad_w, &str_h, &str_w, &ups_x, &ups_y);
    opendnnGetConvolutionGroupCount(conv_desc, &group);

    float *col_buf;
    size_t col_cnt_in_batch = top_h*top_w*bot_c*fil_h*fil_w;
    size_t col_cnt = bot_n*col_cnt_in_batch;
    cudaMalloc((void**)&col_buf, sizeof(float)*col_cnt);

    // Explicitly form column matrix thourgh im2col
    /* if (fil_h != 1){ */
        for (int n = 0; n < bot_n; ++n) {
            float* input = bottom + n*bot_nst;
            float* col_in_batch = col_buf + n*col_cnt_in_batch;
            cudaMemset(col_in_batch, 0, sizeof(float)*col_cnt_in_batch);

            im2col_gpu((float*)input,
                bot_c, bot_w, bot_h, fil_h, fil_w,
                pad_h, pad_w, str_h, str_w,
                1, 1, (float*)col_in_batch);
        }
    /* } else { */
    /*   col_buf = bottom; */
    /* } */

    int fil_out_ = fil_out / group;
    int fil_in_  = fil_in / group;
    int bot_c_   = bot_c / group;
    int top_c_   = top_c / group;

    // Forward through gemm
    float* col_in_batch = col_buf;
    float* output = top;
    for (int g = 0; g < group; g++) {
        const int M = top_c_;
        const int N = top_h*top_w;
        const int K = bot_c_*fil_h*fil_w;
        const float alpha = 1.0f;
        const float beta = 0.0f;
        const int weight_offset = fil_out_*fil_in_*fil_h*fil_w;
        const int col_offset = bot_c_*fil_h*fil_w*top_h*top_w;
        const int output_offset = top_c_*top_h*top_w;

        // Number:: Conversion from native type to NumberADT ===============
        int total = fil_out_ * fil_in_ * fil_h * fil_w;
        Number* f_buf = new Number[total];
        for (int i = 0; i < total; i++) {
            f_buf[i].set(util::to_string(idx));
            f_buf[i] = (filter + weight_offset * g)[i];
        }
        // Number:: Memory allocation ======================================
        Number* f_gpu;
        cudaMalloc(&f_gpu, sizeof(Number)*total);
        cudaMemcpy(f_gpu, f_buf, sizeof(Number)*total, cudaMemcpyHostToDevice);
        // =================================================================

        Number *A = f_gpu; // Number::
        float *B = col_in_batch + col_offset * g;
        float *C = output + output_offset * g;

        const int N_BATCH = top_n;
        dim3 grid((N+BLOCK_SIZE-1)/BLOCK_SIZE, (M+BLOCK_SIZE-1)/BLOCK_SIZE, N_BATCH);
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        matmul_block_lin_shared_batch<<<grid,block>>>(A,B,C,M,K,K,N,M,N,group);

        // Number:: Release memory =========================================
        cudaFree(f_gpu);
        delete [] f_buf;
        // =================================================================
    }

    cudaFree(col_buf);
    idx++;

    DEBUG(bottom);
    DEBUG(top);
}
