#ifdef OPENDNN

#include <cmath>
#include <iostream>
#include <iomanip>
#include "NumberADT/Number.hpp"
extern "C" {
#include "opendnn.h"
}
#include <limits>
#include <string>

#include <sstream>
#include <fstream>
#include <iterator>
#include <vector>

using namespace std;


// Tensor management
void opendnnCreate (opendnnHandle_t* handle) {
    Number::ParseConfigs("config.txt");
    *handle = new opendnnContext;
    (*handle)->driver_num_ = 0;
}

extern "C"
void opendnnCreateTensorDescriptor (opendnnTensorDescriptor_t* tensor_desc) {
    *tensor_desc = new opendnnTensorStruct;
}

extern "C"
void opendnnSetTensor4dDescriptor (opendnnTensorDescriptor_t tens, int n, int c, int h, int w) {
    int w_str = 1;
    int h_str = w * w_str;
    int c_str = h * h_str;
    int n_str = c * c_str;
    opendnnSetTensor4dDescriptorEx (tens, n, c, h, w, n_str, c_str, h_str, w_str);
}

extern "C"
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

extern "C"
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
extern "C"
void opendnnCreateFilterDescriptor (opendnnFilterDescriptor_t* filter) {
    *filter = new opendnnFilterStruct;
}

extern "C"
void opendnnSetFilter4dDescriptor (opendnnFilterDescriptor_t filter, int out, int in, int h, int w) {
    filter->output_ = out;
    filter->input_ = in;
    filter->height_ = h;
    filter->width_ = w;
}

extern "C"
void opendnnGetFilter4dDescriptor (opendnnFilterDescriptor_t filter, int* out, int* in, int* h, int* w) {
    *out = filter->output_;
    *in = filter->input_;
    *h = filter->height_;
    *w = filter->width_;
}


// Convolution
extern "C"
void opendnnCreateConvolutionDescriptor (opendnnConvolutionDescriptor_t* conv_desc) {
    *conv_desc = new opendnnConvolutionStruct;
    (*conv_desc)->group = 1;
}

extern "C"
void opendnnSetConvolution2dDescriptor (opendnnConvolutionDescriptor_t conv_desc, int ph, int pw, int sh, int sw, int ux, int uy) {
    conv_desc->pad_h = ph;
    conv_desc->pad_w = pw;
    conv_desc->vertical_stride = sw;
    conv_desc->horizon_stride = sh;
    conv_desc->upscale_x = ux;
    conv_desc->upscale_y = uy;
}

extern "C"
void opendnnGetConvolution2dDescriptor (opendnnConvolutionDescriptor_t conv_desc,
  int* ph, int* pw, int* sh, int* sw, int* ux, int* uy) {
    *ph = conv_desc->pad_h;
    *pw = conv_desc->pad_w;
    *sw = conv_desc->vertical_stride;
    *sh = conv_desc->horizon_stride;
    *ux = conv_desc->upscale_x;
    *uy = conv_desc->upscale_y;
}

extern "C"
void opendnnSetConvolutionGroupCount(opendnnConvolutionDescriptor_t conv_desc, int group) {
    conv_desc->group = group;
}

extern "C"
void opendnnGetConvolutionGroupCount(opendnnConvolutionDescriptor_t conv_desc, int* group) {
    *group = conv_desc->group;
}


// Pooling
extern "C"
void opendnnCreatePoolingDescriptor (opendnnPoolingDescriptor_t* pool_desc) {
    *pool_desc = new opendnnPoolingStruct;
}

extern "C"
void opendnnSetPooling2dDescriptor (opendnnPoolingDescriptor_t pool_desc, opendnnPoolingMode_t mode,
  int wh, int ww, int vp, int hp, int vs, int hs) {
    pool_desc->pooling_mode = mode;
    pool_desc->w_height = wh;
    pool_desc->w_width = ww;
    pool_desc->vertical_padding = vp;
    pool_desc->horizon_padding = hp;
    pool_desc->vertical_stride = vs;
    pool_desc->horizon_stride = hs;
}

extern "C"
void opendnnGetPooling2dDescriptor (opendnnPoolingDescriptor_t pool_desc, opendnnPoolingMode_t* mode,
  int* wh, int* ww, int* vp, int* hp, int* vs, int* hs) {
    *mode = pool_desc->pooling_mode;
    *wh = pool_desc->w_height;
    *ww = pool_desc->w_width;
    *vp = pool_desc->vertical_padding;
    *hp = pool_desc->horizon_padding;
    *vs = pool_desc->vertical_stride;
    *hs = pool_desc->horizon_stride;
}


// Normalization
extern "C"
void opendnnCreateNormDescriptor (opendnnNormDescriptor_t* norm_desc) {
    *norm_desc = new opendnnNormStruct;
}

extern "C"
void opendnnSetNormDescriptor (opendnnNormDescriptor_t norm_desc, int N, double a, double b, double K, opendnnNormMode_t mode) {
    norm_desc->normN = N;
    norm_desc->normAlpha = a;
    norm_desc->normBeta = b;
    norm_desc->normK = K;
    norm_desc->normMode = mode;
}

extern "C"
void opendnnGetNormDescriptor (opendnnNormDescriptor_t norm_desc, int *N, double *a, double *b, double *K, opendnnNormMode_t* mode) {
    *N = norm_desc->normN;
    *a = norm_desc->normAlpha;
    *b = norm_desc->normBeta;
    *K = norm_desc->normK;
    *mode = norm_desc->normMode;
}


// Computation of Pooling
extern "C"
void opendnnPoolingForward (opendnnHandle_t handle, opendnnPoolingDescriptor_t pool_desc,
  opendnnTensorDescriptor_t bottom_desc, const Dtype* bottom, opendnnTensorDescriptor_t top_desc, Dtype* top) {
    opendnnPoolingMode_t mode;
    int bot_n, bot_c, bot_h, bot_w, bot_nst, bot_cst, bot_hst, bot_wst;
    int top_n, top_c, top_h, top_w, top_nst, top_cst, top_hst, top_wst;
    int kernel_h, kernel_w, v_pad, h_pad, v_str, h_str;
    opendnnGetPooling2dDescriptor (pool_desc, &mode, &kernel_h, &kernel_w, &v_pad, &h_pad, &v_str, &h_str);
    opendnnGetTensor4dDescriptor (bottom_desc, &bot_n, &bot_c, &bot_h, &bot_w,
        &bot_nst, &bot_cst, &bot_hst, &bot_wst);
    opendnnGetTensor4dDescriptor (top_desc, &top_n, &top_c, &top_h, &top_w,
        &top_nst, &top_cst, &top_hst, &top_wst);

    Dtype* top_ = top;
    const Dtype* bottom_ = bottom;

    if (mode == POOLING_MAX) {
      for (int i = 0; i < top_n*top_c*top_h*top_w; ++i) {
        // top[i] = Dtype(-3.402823e+38);
        top_[i] = -std::numeric_limits<float>::infinity();
      }
      for (int i = 0; i < top_n; ++i) {
        for (int j = 0; j < top_c; ++j) {
          for (int k = 0; k < top_h; ++k) {
            for (int l = 0; l < top_w; ++l) {
              int hstart = k*h_str - h_pad;
              int wstart = l*v_str - v_pad;
              int hend = std::min(hstart + kernel_h, bot_h);
              int wend = std::min(wstart + kernel_w, bot_w);
              hstart = std::max(hstart, 0);
              wstart = std::max(wstart, 0);
              const int pool_index = k*top_w + l;
              for (int m = hstart; m < hend; ++m) {
                for (int n = wstart; n < wend; ++n) {
                  const int index = m * bot_w + n;
                  if (bottom_[index] > top_[pool_index]) {
                    top_[pool_index] = bottom_[index];
                  }
                }
              }
            }
          }
          // compute offset
          bottom_ += bot_cst;
          top_ += top_cst;
        }
      }
    }
    else if (mode == POOLING_AVG) {
      for (int i = 0; i < top_n*top_c*top_h*top_w; ++i) {
        top_[i] = 0;
      }
      for (int i = 0; i < top_n; ++i) {
        for (int j = 0; j < top_c; ++j) {
          for (int k = 0; k < top_h; ++k) {
            for (int l = 0; l < top_w; ++l) {
              int hstart = k * h_str - h_pad;
              int wstart = l * v_str - v_pad;
              int hend = std::min(hstart + kernel_h, bot_h + h_pad);
              int wend = std::min(wstart + kernel_w, bot_w + v_pad);
              int pool_size = (hend - hstart) * (wend - wstart);
              hstart = std::max(hstart, 0);
              wstart = std::max(wstart, 0);
              hend = std::min(hend, bot_h);
              wend = std::min(wend, bot_w);
              for (int m = hstart; m < hend; ++m) {
                for (int n = wstart; n < wend; ++n) {
                  top_[k * top_w + l] += bottom_[m * bot_w + n];
                }
              }
              top_[k * top_w + l] /= pool_size;
            }
          }
          bottom_ += bot_cst;
          top_ += top_cst;
        }
      }
    }
    DEBUG(bottom);
    DEBUG(top);
}

extern "C"
void opendnnNormForward (opendnnHandle_t, opendnnNormDescriptor_t norm_desc,
  opendnnTensorDescriptor_t bottom_tens, const Dtype* bottom,
  opendnnTensorDescriptor_t top_tens, Dtype* top) {
    int local_size;
    double alpha, beta, K;
    int bot_n, bot_c, bot_h, bot_w, bot_nst, bot_cst, bot_hst, bot_wst;
    int top_n, top_c, top_h, top_w, top_nst, top_cst, top_hst, top_wst;
    opendnnNormMode_t mode;
    opendnnGetNormDescriptor(norm_desc, &local_size, &alpha, &beta, &K, &mode);
    opendnnGetTensor4dDescriptor(bottom_tens, &bot_n, &bot_c, &bot_h, &bot_w,
        &bot_nst, &bot_cst, &bot_hst, &bot_wst);
    opendnnGetTensor4dDescriptor(top_tens, &top_n, &top_c, &top_h, &top_w,
        &top_nst, &top_cst, &top_hst, &top_wst);

    if (mode == CROSS_CHANNEL) {
      for (int n = 0; n < bot_n; ++n) {
        for (int c = 0; c < bot_c; ++c) {
          for (int h = 0; h < bot_h; ++h) {
            for (int w = 0; w < bot_w; ++w) {
              int start = c - (local_size - 1) / 2;
              int end = std::min (start + local_size, bot_c);
              start = std::max (start, 0);
              Dtype scale = K;
              for (int i = start; i < end; ++i) {
                  Dtype value = *(bottom + w * bot_wst + h * bot_hst + i * bot_cst + n * bot_nst);
                  scale += (value * value * alpha) / local_size;
              }
              *(top + w * top_wst + h * top_hst + c * top_cst + n * top_nst) = 
                  *(bottom + w * bot_wst + h * bot_hst + c * bot_cst + n * bot_nst) / pow (scale, beta);
            }
          }
        }
      }
    }
    else if (mode == WITHIN_CHANNEL) {
      for (int n = 0; n < bot_n; ++n) {
        for (int c = 0; c < bot_c; ++c) {
          for (int h = 0; h < bot_h; ++h) {
            int h_start = h - (local_size - 1) / 2;
            int h_end = std::min (h_start + local_size, bot_h);
            h_start = std::max (h_start, 0);
            for (int w = 0; w < bot_w; ++w) {
              Dtype scale = K;
              int w_start = w - (local_size - 1) / 2;
              int w_end = std::min(w_start + local_size, bot_w);
              w_start = std::max(w_start, 0);
              for (int nh = h_start; nh < h_end; ++nh) {
                for (int nw = w_start; nw < w_end; ++nw) {
                  Dtype value = bottom[nw*bot_wst + nh*bot_hst + c*bot_cst + n*bot_nst];
                  scale += (value * value * alpha) / (local_size * local_size);
                }
              }
              top[w*top_wst + h*top_hst + c*top_cst + n*top_nst] =
                  bottom[w*bot_wst + h*bot_hst + c*bot_cst + n*bot_nst] /
                  std::pow(scale, beta);
            }
          }
        }
      }
    }
    DEBUG(bottom);
    DEBUG(top);
}


// Convolution computation API
extern "C"
void opendnnAddTensor (opendnnHandle_t handle,
  opendnnTensorDescriptor_t bias_, const Dtype* bias_data, opendnnTensorDescriptor_t top_, Dtype* top_data) {
    int bias_n, bias_c, bias_h, bias_w, bias_nst, bias_cst, bias_hst, bias_wst;    
    int top_n, top_c, top_h, top_w, top_nst, top_cst, top_hst, top_wst;
    opendnnGetTensor4dDescriptor (bias_, &bias_n, &bias_c, &bias_h, &bias_w,
        &bias_nst, &bias_cst, &bias_hst, &bias_wst);
    opendnnGetTensor4dDescriptor (top_, &top_n, &top_c, &top_h, &top_w,
        &top_nst, &top_cst, &top_hst, &top_wst);

#ifdef __OPENDNN_DEBUG__
    std::cout << "Add bias\n";
#endif

    DEBUG(top_data);
    for (int i = 0; i < top_n; ++i) {
      for (int j = 0; j < top_c; ++j) {
        for (int k = 0; k < top_h; ++k) {
          for (int l = 0; l < top_w; ++l) {
            top_data[l*top_wst + k*top_hst + j*top_cst + i*top_nst] += bias_data[j*bias_cst];
          }
        }
      }
    }
    DEBUG(top_data);
}


/* void opendnnConvolutionForward (opendnnHandle_t handle, */
/*   opendnnTensorDescriptor_t bottom_desc, const Dtype* bottom, */
/*   opendnnFilterDescriptor_t filter_desc, const Dtype* filter, */
/*   opendnnConvolutionDescriptor_t conv_desc, */
/*   opendnnTensorDescriptor_t top_desc, Dtype* top) { */
/*     int bot_n, bot_c, bot_h, bot_w, bot_nst, bot_cst, bot_hst, bot_wst; */
/*     int top_n, top_c, top_h, top_w, top_nst, top_cst, top_hst, top_wst; */
/*     int pad_h, pad_w, str_h, str_w, ups_x, ups_y; */
/*     int fil_out, fil_in, fil_h, fil_w; */
/*     int group, b_offset, t_offset, f_offset; */
/*     opendnnGetTensor4dDescriptor (bottom_desc, &bot_n, &bot_c, &bot_h, &bot_w, */
/*         &bot_nst, &bot_cst, &bot_hst, &bot_wst); */
/*     opendnnGetTensor4dDescriptor (top_desc, &top_n, &top_c, &top_h, &top_w, */
/*         &top_nst, &top_cst, &top_hst, &top_wst); */
/*     opendnnGetFilter4dDescriptor (filter_desc, &fil_out, &fil_in, &fil_h, &fil_w); */
/*     opendnnGetConvolution2dDescriptor (conv_desc, &pad_h, &pad_w, &str_h, &str_w, &ups_x, &ups_y); */
/*     opendnnGetConvolutionGroupCount(conv_desc, &group); */
/*  */
/*     for (int n = 0; n < top_n; n++) { */
/*       for (int row = 0; row < top_h; row++) { */
/*         for (int col = 0; col < top_w; col++) { */
/*           for (int to = 0; to < top_c; to++) { */
/*             top[row*top_wst + col*top_hst + to*top_cst + n*top_nst] = 0; */
/*     }}}} */
/*  */
/*     fil_out = fil_out / group; */
/*     fil_in = fil_in / group; */
/*     bot_c = bot_c / group; */
/*     top_c = top_c / group; */
/*  */
/*     // TODO: effective channel, now process group sequentially. Any better apporaches? */
/*     f_offset = fil_out*fil_in*fil_h*fil_w; */
/*     b_offset = bot_c*bot_h*bot_w; */
/*     t_offset = top_c*top_h*top_w; */
/*  */
/*     for (int g = 0; g < group; ++g) { */
/*       for (int n = 0; n < top_n; ++n) {  // TODO: batch processing */
/*         for (int c = 0; c < top_c; ++c) { */
/*           for (int h = 0; h < top_h; ++h) { */
/*             for (int w = 0; w < top_w; ++w) { */
/*               float sum = 0.0; */
/*               for (int k = 0; k < bot_c; ++k) { */
/*                 for (int fh = 0; fh < fil_h; ++fh) { */
/*                   for (int fw = 0; fw < fil_w; fw++) { */
/*                     int in_h = h*str_h - pad_h + fh; */
/*                     int in_w = w*str_w - pad_w + fw; */
/*                     if (in_w >= 0 && in_w < bot_w && in_h >= 0 && in_h < bot_h) { */
/*                         sum += (bottom+b_offset*g)[in_w*bot_wst + in_h*bot_hst + k*bot_cst + n*bot_nst] * */
/*                                (filter+f_offset*g)[fw + fh*fil_w + k*fil_w*fil_h + c*fil_in*fil_w*fil_h]; */
/*                     } */
/*                   } */
/*                 } */
/*               } */
/*               (top+t_offset*g)[w*top_wst + h*top_hst + c*top_cst + n*top_nst] = sum; */
/*             } */
/*           } */
/*         } */
/*       } */
/*     } */
/*     DEBUG(bottom); */
/*     DEBUG(top); */
/* } */

#endif //OPENDNN
