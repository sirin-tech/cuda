#include "common.h"
#include <tuple>

ETERM *GetDeviceCount() {
  int devCount;
  cudaError_t result = cudaGetDeviceCount(&devCount);
  switch (result) {
    case cudaSuccess: return erl_format("{~a,~i}", "ok", devCount);
    case cudaErrorNoDevice: return erl_format("{~a,~i}", "ok", 0);
    default: return CudaError(result);
  }
}

ETERM *GetMemory() {
  size_t freeMem, totalMem;
  cudaError_t result = cudaMemGetInfo(&freeMem, &totalMem);
  if (result != cudaSuccess) return CudaError(result);
  return erl_format("{~a,{~i,~i}}", "ok", freeMem, totalMem);
}

ETERM *GetDriverVersion() {
  int version;
  cudaError_t result = cudaDriverGetVersion(&version);
  if (result != cudaSuccess) return CudaError(result);
  return erl_format("{~a,~i}", "ok", version);
}

ETERM *GetRuntimeVersion() {
  int version;
  cudaError_t result = cudaRuntimeGetVersion(&version);
  if (result != cudaSuccess) return CudaError(result);
  return erl_format("{~a,~i}", "ok", version);
}

ETERM *Info(ETERM *arg) {
  if (IS_NIL(arg)) {
    auto deviceCount = GetDeviceCount();
    if (!IS_OK_TUPLE(deviceCount)) return deviceCount;
    auto memory = GetMemory();
    if (!IS_OK_TUPLE(memory)) return memory;
    auto driverVersion = GetDriverVersion();
    if (!IS_OK_TUPLE(driverVersion)) return driverVersion;
    auto runtimeVersion = GetRuntimeVersion();
    if (!IS_OK_TUPLE(runtimeVersion)) return runtimeVersion;
    return erl_format("[{~a,~w},{~a,~w},{~a,~w},{~a,~w}]",
      "device_count", erl_element(2, deviceCount),
      "driver_version", erl_element(2, driverVersion),
      "memory", erl_element(2, memory),
      "runtime_version", erl_element(2, runtimeVersion));
  } else if (ERL_IS_ATOM(arg)) {
    if (ATOM_EQ(arg, "device_count")) return GetDeviceCount();
    else if (ATOM_EQ(arg, "driver_version")) return GetDriverVersion();
    else if (ATOM_EQ(arg, "memory")) return GetMemory();
    else if (ATOM_EQ(arg, "runtime_version")) return GetRuntimeVersion();
  }
  return ERROR("bad argument");
}
