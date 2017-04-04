#include <vector>
#include "common.h"
#include "driver.h"

Driver::Driver(int deviceNo) {
  CUresult result = CUDA_SUCCESS;
  result = cuDeviceGet(&device, deviceNo);
  if (result != CUDA_SUCCESS) throw DriverError(result);
  result = cuCtxCreate(&context, 0, device);
  if (result != CUDA_SUCCESS) throw DriverError(result);
  DEBUG("Driver initialized for device #" << deviceNo);
}

Driver::~Driver() {
  for (auto module = modules.begin(); module != modules.end(); ++module) {
    cuModuleUnload(module->second);
  }
  for (auto mem = memory.begin(); mem != memory.end(); ++mem) {
    delete mem->second;
  }
  cuCtxDestroy(context);
  DEBUG("Driver destroyed");
}

int Driver::Compile(std::list<std::string> sources, LinkerOptions &options) {
  Linker linker(options);
  linker.Run(sources);

  CUmodule module;
  auto result = cuModuleLoadData(&module, linker.cubin);
  if (result != CUDA_SUCCESS) throw DriverError(result);
  int moduleNo = modules.size() + 1;
  modules.insert(std::pair<int, CUmodule>(moduleNo, module));

  return moduleNo;
}

int Driver::LoadModule(std::string cubin, LinkerOptions &options) {
  Linker linker(options);
  CUmodule module;
  auto result = cuModuleLoadDataEx(&module, cubin.c_str(), linker.OptionsSize(),
                                   linker.OptionsKeys(), linker.OptionsValues());
  if (result != CUDA_SUCCESS) throw DriverError(result);
  int moduleNo = modules.size() + 1;
  modules.insert(std::pair<int, CUmodule>(moduleNo, module));
  return moduleNo;
}


int Driver::LoadMemory(const void *src, size_t size) {
  DeviceMemory *mem = new DeviceMemory(src, size);
  int memNo = memory.size() + 1;
  memory.insert(std::pair<int, DeviceMemory *>(memNo, mem));
  return memNo;
}

void Driver::UnloadMemory(int id) {
  auto mem = memory.find(id);
  if (mem == memory.end()) throw StringError("Invalid memory handle");
  delete mem->second;
  memory.erase(id);
}

void Driver::ReadMemory(int id, void *dst, int size) {
  auto mem = memory.find(id);
  if (mem == memory.end()) throw StringError("Invalid memory handle");
  mem->second->Read(dst, size);
}

int Driver::GetMemorySize(int id) {
  auto mem = memory.find(id);
  if (mem == memory.end()) return -1;
  return mem->second->GetSize();
}

DeviceMemory *Driver::GetMemory(int id) {
  auto mem = memory.find(id);
  if (mem == memory.end()) return NULL;
  return mem->second;
}

void Driver::Run(int moduleNo, std::string funcName, int gx, int gy, int gz,
                 int bx, int by, int bz, RunParameters &params) {
  auto module = modules.find(moduleNo);
  if (module == modules.end()) throw StringError("Invalid module handle");
  CUfunction func;
  auto result = cuModuleGetFunction(&func, module->second, funcName.c_str());
  if (result != CUDA_SUCCESS) throw DriverError(result);
  // void **paramsPtr = params.empty() ? NULL : args.data();

  result = cuLaunchKernel(func, gx, gy, gz, bx, by, bz, 0, 0, params.GetPtr(), 0);
  if (result != CUDA_SUCCESS) throw DriverError(result, "Driver:execution");
}

template <> DeviceMemory *Driver::Unpack<DeviceMemory *>(ETERM *value) {
  if (!ERL_IS_TUPLE(value) || erl_size(value) != 2) {
    throw StringError("Invalid memory handle");
  }
  auto a = erl_element(1, value);
  auto v = erl_element(2, value);
  if (!ERL_IS_ATOM(a) || !ATOM_EQ(a, "memory")) {
    throw StringError("Invalid memory handle");
  }
  auto mem = GetMemory(Get<int>(v));
  if (!mem) throw StringError("Invalid memory handle");
  return mem;
}

template <> CUmodule Driver::Unpack<CUmodule>(ETERM *value) {
  if (!ERL_IS_TUPLE(value) || erl_size(value) != 2) {
    throw StringError("Invalid module handle");
  }
  auto a = erl_element(1, value);
  auto v = erl_element(2, value);
  if (!ERL_IS_ATOM(a) || !ATOM_EQ(a, "module")) {
    throw StringError("Invalid module handle");
  }
  auto module = modules.find(Get<int>(v));
  if (module == modules.end())  throw StringError("Invalid memory handle");
  return module->second;
}

ETERM *Driver::PackMemory(int idx) {
  return FORMAT("{~a,~i}", C_STR("memory"), idx);
}

ETERM *Driver::PackModule(int idx) {
  return FORMAT("{~a,~i}", C_STR("module"), idx);
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

size_t Linker::OptionsSize() {
  if (!initialized) throw StringError("Unintialized linker used");
  return optKeys.size();
}

CUjit_option *Linker::OptionsKeys() {
  if (!initialized) throw StringError("Unintialized linker used");
  return optKeys.data();
}

void **Linker::OptionsValues() {
  if (!initialized) throw StringError("Unintialized linker used");
  return optValues.data();
}

void Linker::Run(std::list<std::string> sources) {
  if (!initialized) throw StringError("Unintialized linker used");

  CUresult result;

  result = cuLinkCreate(optKeys.size(), optKeys.data(), optValues.data(), &state);
  if (result != CUDA_SUCCESS) throw DriverError(result);

  for (auto it = std::begin(sources); it != std::end(sources); ++it) {
    result = cuLinkAddData(state, CU_JIT_INPUT_PTX, (void *)it->c_str(), it->size() + 1, 0, 0, 0, 0);
    if (result != CUDA_SUCCESS) throw CompileError(result, infoLog, errorLog);
  }

  result = cuLinkComplete(state, &cubin, &cubinSize);
  if (result != CUDA_SUCCESS) throw CompileError(result, infoLog, errorLog);
}
