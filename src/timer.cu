#include <boost/date_time/posix_time/posix_time.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>  // pair
#include <vector>

#include "timer.hpp"

Timer::Timer()
    : initted_(false),
      running_(false),
      has_run_at_least_once_(false) {
  Init();
}

Timer::~Timer() {
#ifndef CPU_ONLY
    CUDA_CHECK(cudaEventDestroy(start_gpu_));
    CUDA_CHECK(cudaEventDestroy(stop_gpu_));
#else
    NO_GPU;
#endif
}

void Timer::Start() {
  if (!running()) {
#ifndef CPU_ONLY
      CUDA_CHECK(cudaEventRecord(start_gpu_, 0));
#else
      NO_GPU;
#endif
    running_ = true;
    has_run_at_least_once_ = true;
  }
}

void Timer::Stop() {
  if (running()) {
#ifndef CPU_ONLY
      CUDA_CHECK(cudaEventRecord(stop_gpu_, 0));
      CUDA_CHECK(cudaEventSynchronize(stop_gpu_));
#else
      NO_GPU;
#endif
    running_ = false;
  }
}


float Timer::MicroSeconds() {
  if (!has_run_at_least_once()) {
    return 0;
  }
  if (running()) {
    Stop();
  }
#ifndef CPU_ONLY
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_milliseconds_, start_gpu_,
                                    stop_gpu_));
    // Cuda only measure milliseconds
    elapsed_microseconds_ = elapsed_milliseconds_ * 1000;
#else
      NO_GPU;
#endif
  return elapsed_microseconds_;
}

float Timer::MilliSeconds() {
  if (!has_run_at_least_once()) {
    return 0;
  }
  if (running()) {
    Stop();
  }
#ifndef CPU_ONLY
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_milliseconds_, start_gpu_,
                                    stop_gpu_));
#else
      NO_GPU;
#endif
  return elapsed_milliseconds_;
}

float Timer::Seconds() {
  return MilliSeconds() / 1000.;
}

void Timer::Init() {
  if (!initted()) {
#ifndef CPU_ONLY
      CUDA_CHECK(cudaEventCreate(&start_gpu_));
      CUDA_CHECK(cudaEventCreate(&stop_gpu_));
#else
      NO_GPU;
#endif
    initted_ = true;
  }
}
