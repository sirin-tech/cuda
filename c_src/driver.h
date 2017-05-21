#ifndef __DRIVER_H__
#define __DRIVER_H__

#include <vector>
#include <memory>

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
  size_t OptionsSize();
  CUjit_option *OptionsKeys();
  void **OptionsValues();
};

typedef std::tuple<CUipcMemHandle, size_t> SharedMemory;

class DeviceMemory {
private:
  CUdeviceptr ptr = (CUdeviceptr)NULL;
  bool initialized = false;
  bool shared = false;
  size_t size;
public:
  DeviceMemory(const void *src, size_t srcSize): size(srcSize) {
    auto result = cuMemAlloc(&ptr, size);
    if (result != CUDA_SUCCESS) throw DriverError(result, "DeviceMemory:allocate");
    result = cuMemcpyHtoD(ptr, src, size);
    if (result != CUDA_SUCCESS) throw DriverError(result, "DeviceMemory:copy");
    initialized = true;
    DEBUG("Device memory initialized with size " << srcSize);
  }
  DeviceMemory(CUipcMemHandle handle, size_t memSize): size(memSize) {
    auto result = cuIpcOpenMemHandle(&ptr, handle, CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS);
    if (result != CUDA_SUCCESS) throw DriverError(result, "DeviceMemory:ipc");
    shared = true;
    initialized = true;
    DEBUG("Device memory initialized from shared with size " << memSize);
  }

  ~DeviceMemory() {
    DEBUG("Device memory destroyed");
    if (initialized) {
      shared ? cuIpcCloseMemHandle(ptr) : cuMemFree(ptr);
    }
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

class RunArguments {
private:
  std::vector<void *> values;
public:
  ~RunArguments() {
    DEBUG("Run arguments destroyed");
    for (auto it = values.begin(); it != values.end(); ++it) std::free(*it);
    values.clear();
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

typedef std::tuple<std::string, int, int, int, int, int, int> RunParameters;
typedef std::tuple<RunParameters, std::shared_ptr<RunArguments>> RunEnvironment;

class Driver {
private:
  CUdevice    device;
  CUcontext   context;
  std::map<int, CUmodule> modules;
  std::map<int, DeviceMemory *> memory;
public:
  Driver(int deviceNo);
  ~Driver();
  CUdevice GetHandle() { return device; }
  int Compile(std::list<std::string> sources, LinkerOptions &options);
  int LoadModule(std::string cubin, LinkerOptions &options);
  CUmodule GetModule(int id);
  int LoadMemory(const void *src, size_t size);
  int LoadMemory(SharedMemory mem);
  void UnloadMemory(int id);
  void ReadMemory(int id, void *dst, int size = -1);
  int GetMemorySize(int id);
  DeviceMemory *GetMemory(int id);
  SharedMemory ShareMemory(int id);
  void Run(int moduleNo, RunParameters &params, std::shared_ptr<RunArguments> &args);
  void Stream(int moduleNo, std::vector<RunEnvironment> &batch);
  template <typename T> T Unpack(ETERM *value);
  ETERM *PackMemory(int idx);
  ETERM *PackMemory(SharedMemory mem);
  ETERM *PackModule(int idx);
};

#endif // __DRIVER_H__
