#include "driver.h"

#define ASSERT_SUCCESS(func) \
  if (func != CUDA_SUCCESS) throw new Error(CudaDriverError(result))

Driver::Driver(int deviceNo) {
  CUresult result = CUDA_SUCCESS;
  ASSERT_SUCCESS(cuDeviceGet(&device, deviceNo));
  ASSERT_SUCCESS(cuCtxCreate(&context, 0, device));
  cuCtxPopCurrent(&context);
}

Driver::~Driver() {
  cuCtxPopCurrent(&context);
}

// void Driver::Execute(std::list<std::string> sources) {}
