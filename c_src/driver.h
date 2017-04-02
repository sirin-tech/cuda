#ifndef __DRIVER_H__
#define __DRIVER_H__

#include <vector>

#include "common.h"

#define LINKER_BUFFER_SIZE 8192

struct LinkerOptions {
  int maxRegisters;
  int threadsPerBlock;
  int optimizationLevel;
  int target;
  int debug;
  int verbose;
  int infoSize;
  int errorSize;
};

class Linker {
private:
  size_t cubinSize = 0;
  float walltime = 0.0;
  unsigned int threadsPerBlock = 0;
  std::vector<CUjit_option> optKeys;
  std::vector<void *> optValues;
  CUlinkState state;
  char *infoLog;
  char *errorLog;
  bool initialized = false;
public:
  void *cubin = NULL;
  Linker(LinkerOptions &options);
  ~Linker();
  void Run(std::list<std::string> sources);
};

class DeviceMemory {
private:
  CUdeviceptr ptr = (CUdeviceptr)NULL;
  bool initialized = false;
  size_t size;
public:
  DeviceMemory(const void *src, size_t srcSize): size(srcSize) {
    auto result = cuMemAlloc(&ptr, size);
    if (result != CUDA_SUCCESS) throw DriverError(result, "DeviceMemory:allocate");
    result = cuMemcpyHtoD(ptr, src, size);
    if (result != CUDA_SUCCESS) throw DriverError(result, "DeviceMemory:copy");
    initialized = true;
  }

  ~DeviceMemory() {
    if (initialized) cuMemFree(ptr);
  }

  void Read(void *dst, int dstSize = -1) {
    if (dstSize < 0) dstSize = size;
    auto r = cudaMemcpy(dst, (void *)ptr, dstSize, cudaMemcpyDeviceToHost);
    if (r != cudaSuccess) throw RuntimeError(r, "DeviceMemory:read");
  }

  size_t GetSize() {
    return size;
  }

  CUdeviceptr GetPtr() {
    return ptr;
  }

  CUdeviceptr *GetPtrPtr() {
    return &ptr;
  }
};

class RunParameters {
private:
  std::vector<void *> values;
public:
  ~RunParameters() {
    for (auto it = values.begin(); it != values.end(); ++it) std::free(*it);
  }

  template <typename T> void Add(T param) {
    auto ptr = (T *)malloc(sizeof(T));
    *ptr = param;
    values.push_back(ptr);
  }

  void Add(DeviceMemory &memory) {
    auto ptr = (CUdeviceptr *)malloc(sizeof(CUdeviceptr));
    *ptr = memory.GetPtr();
    values.push_back(ptr);
  }

  void **GetPtr() {
    if (values.empty()) return NULL;
    return values.data();
  }
};

/*
class Worker {
public:
  Worker(CUmodule &module, std::string func, std::vector<void *>params);
  ~Worker();
  void Run();
};*/

class CompileError : public Error {
public:
  CUresult code;
  std::string infoLog;
  std::string errorLog;
  CompileError(CUresult errorNo, char *info, char *error) :
    Error(),
    code(errorNo),
    infoLog(info, strlen(info)),
    errorLog(error, strlen(error)) {}
  virtual ETERM *AsTerm();
};

class Driver {
private:
  CUdevice    device;
  CUcontext   context;
  std::map<int, CUmodule> modules;
  std::map<int, DeviceMemory *> memory;
public:
  Driver(int deviceNo);
  ~Driver();
  int Compile(std::list<std::string> sources, LinkerOptions &options);
  int LoadMemory(const void *src, size_t size);
  void UnloadMemory(int id);
  void ReadMemory(int id, void *dst, int size = -1);
  int GetMemorySize(int id);
  DeviceMemory *GetMemory(int id);
  void Run(int moduleNo, std::string funcName, int gx, int gy, int gz,
           int bx, int by, int bz, RunParameters &params);
  template <typename T> T Unpack(ETERM *value);
  ETERM *PackMemory(int idx);
  ETERM *PackModule(int idx);
  // void Run(int module, std::string func, std::vector<void *> args);
};

#endif // __DRIVER_H__
