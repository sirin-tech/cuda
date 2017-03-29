#include <vector>
#include "common.h"
#include "driver.h"

Driver::Driver(int deviceNo) {
  CUresult result = CUDA_SUCCESS;
  result = cuDeviceGet(&device, deviceNo);
  if (result != CUDA_SUCCESS) throw DriverError(result);
  result = cuCtxCreate(&context, 0, device);
  if (result != CUDA_SUCCESS) throw DriverError(result);
}

Driver::~Driver() {
  for (auto module = modules.begin(); module != modules.end(); ++module) {
    cuModuleUnload(module->second);
  }
  cuCtxDestroy(context);
  // cuCtxPopCurrent(&context);
}

int Driver::Compile(std::list<std::string> sources, LinkerOptions &options) {
  auto linker = new Linker(options);
  linker->Run(sources);

  CUmodule module;
  auto result = cuModuleLoadData(&module, linker->cubin);
  if (result != CUDA_SUCCESS) throw DriverError(result);
  int moduleNo = modules.size() + 1;
  modules.insert(std::pair<int, CUmodule>(moduleNo, module));

  return moduleNo;
}

ETERM *CompileError::AsTerm() {
  const char *buf;
  auto err = cuGetErrorString(code, &buf);
  ETERM *errStr = err == CUDA_SUCCESS ?
                  erl_mk_binary(buf, strlen(buf)) :
                  MAKE_BINARY("");
  return FORMAT("{~a,~w,~w,~w}", ERROR_STR,
    errStr,
    erl_mk_binary(infoLog.c_str(), infoLog.size()),
    erl_mk_binary(errorLog.c_str(), errorLog.size())
  );
}

Linker::Linker(LinkerOptions &options) {
  if (options.threadsPerBlock >= 0 && options.target >= 0) {
    throw StringError("threads_per_block linker option can not be used together with target option");
  }

  int infoSize = options.infoSize > 0 ? options.infoSize : LINKER_BUFFER_SIZE;
  int errorSize = options.errorSize > 0 ? options.errorSize : LINKER_BUFFER_SIZE;
  infoLog = new char[infoSize];
  errorLog = new char[errorSize];

  optKeys.push_back(CU_JIT_WALL_TIME);
  optValues.push_back((void *)&walltime);
  optKeys.push_back(CU_JIT_INFO_LOG_BUFFER);
  optValues.push_back((void *)infoLog);
  optKeys.push_back(CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES);
  optValues.push_back((void *)(intptr_t)infoSize);
  optKeys.push_back(CU_JIT_ERROR_LOG_BUFFER);
  optValues.push_back((void *)errorLog);
  optKeys.push_back(CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES);
  optValues.push_back((void *)(intptr_t)errorSize);
  if (options.maxRegisters >= 0) {
    optKeys.push_back(CU_JIT_MAX_REGISTERS);
    optValues.push_back((void *)&options.maxRegisters);
  }
  if (options.optimizationLevel >= 0) {
    optKeys.push_back(CU_JIT_OPTIMIZATION_LEVEL);
    optValues.push_back((void *)&options.optimizationLevel);
  }
  if (options.threadsPerBlock >= 0) {
    threadsPerBlock = options.threadsPerBlock;
    optKeys.push_back(CU_JIT_THREADS_PER_BLOCK);
    optValues.push_back((void *)&threadsPerBlock);
  } else if (options.target >= 0) {
    optKeys.push_back(CU_JIT_TARGET);
    optValues.push_back((void *)&options.target);
  }
  if (options.debug >= 0) {
    optKeys.push_back(CU_JIT_GENERATE_DEBUG_INFO);
    optValues.push_back((void *)&options.debug);
  }
  if (options.verbose >= 0) {
    optKeys.push_back(CU_JIT_LOG_VERBOSE);
    optValues.push_back((void *)&options.verbose);
  }

  auto result = cuLinkCreate(optKeys.size(), optKeys.data(), optValues.data(), &state);
  if (result != CUDA_SUCCESS) throw DriverError(result);
  initialized = true;
}

Linker::~Linker() {
  if (infoLog) delete infoLog;
  if (errorLog) delete errorLog;
  if (initialized) {
    auto result = cuLinkDestroy(state);
    if (result != CUDA_SUCCESS) throw DriverError(result);
  }
}

void Linker::Run(std::list<std::string> sources) {
  CUresult result;
  if (!initialized) throw StringError("Unintialized linker used");

  for (auto it = std::begin(sources); it != std::end(sources); ++it) {
    result = cuLinkAddData(state, CU_JIT_INPUT_PTX, (void *)it->c_str(), it->size() + 1, 0, 0, 0, 0);
    if (result != CUDA_SUCCESS) throw CompileError(result, infoLog, errorLog);
  }

  result = cuLinkComplete(state, &cubin, &cubinSize);
  if (result != CUDA_SUCCESS) throw CompileError(result, infoLog, errorLog);
}
