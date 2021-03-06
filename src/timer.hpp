#ifndef CAFFE_UTIL_BENCHMARK_H_
#define CAFFE_UTIL_BENCHMARK_H_

#include <boost/date_time/posix_time/posix_time.hpp>

// #include "caffe/util/device_alternate.hpp"

// CUDA: various checks for different function calls.
#define CUDA_CHECK(txt) txt
#define CUBLAS_CHECK(txt) txt

// #define CURAND_CHECK(txt) txt

class Timer {
 public:
  Timer();
  virtual ~Timer();
  virtual void Start();
  virtual void Stop();
  virtual float MilliSeconds();
  virtual float MicroSeconds();
  virtual float Seconds();

  inline bool initted() { return initted_; }
  inline bool running() { return running_; }
  inline bool has_run_at_least_once() { return has_run_at_least_once_; }

 protected:
  void Init();

  bool initted_;
  bool running_;
  bool has_run_at_least_once_;
  cudaEvent_t start_gpu_;
  cudaEvent_t stop_gpu_;
  float elapsed_milliseconds_;
  float elapsed_microseconds_;
};

#endif   // CAFFE_UTIL_BENCHMARK_H_
