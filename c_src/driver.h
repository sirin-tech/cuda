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
public:
  Driver(int deviceNo);
  ~Driver();
  int Compile(std::list<std::string> sources, LinkerOptions &options);
  // void Run(int module, std::string func, std::vector<void *> args);
};

#endif // __DRIVER_H__
