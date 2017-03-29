#include "erlang_port.h"
#include <tuple>

ETERM *GetDeviceCount() {
  int devCount;
  cudaError_t result = cudaGetDeviceCount(&devCount);
  switch (result) {
    case cudaSuccess: return FORMAT("{~a,~i}", OK_STR, devCount);
    case cudaErrorNoDevice: return FORMAT("{~a,~i}", OK_STR, 0);
    default: throw RuntimeError(result);
  }
}

ETERM *GetMemory() {
  size_t freeMem, totalMem;
  cudaError_t result = cudaMemGetInfo(&freeMem, &totalMem);
  if (result != cudaSuccess) throw RuntimeError(result);
  return FORMAT("{~a,{~i,~i}}", OK_STR, freeMem, totalMem);
}

ETERM *GetDriverVersion() {
  int version;
  cudaError_t result = cudaDriverGetVersion(&version);
  if (result != cudaSuccess) throw RuntimeError(result);
  return FORMAT("{~a,~i}", OK_STR, version);
}

ETERM *GetRuntimeVersion() {
  int version;
  cudaError_t result = cudaRuntimeGetVersion(&version);
  if (result != cudaSuccess) throw RuntimeError(result);
  return FORMAT("{~a,~i}", OK_STR, version);
}

ETERM *Info(ErlangPort *, ETERM *arg) {
  if (IS_NIL(arg)) {
    auto deviceCount = GetDeviceCount();
    auto memory = GetMemory();
    auto driverVersion = GetDriverVersion();
    auto runtimeVersion = GetRuntimeVersion();
    return FORMAT("[{~a,~w},{~a,~w},{~a,~w},{~a,~w}]",
      C_STR("device_count"), erl_element(2, deviceCount),
      C_STR("driver_version"), erl_element(2, driverVersion),
      C_STR("memory"), erl_element(2, memory),
      C_STR("runtime_version"), erl_element(2, runtimeVersion));
  } else if (ERL_IS_ATOM(arg)) {
    if (ATOM_EQ(arg, "device_count")) return GetDeviceCount();
    else if (ATOM_EQ(arg, "driver_version")) return GetDriverVersion();
    else if (ATOM_EQ(arg, "memory")) return GetMemory();
    else if (ATOM_EQ(arg, "runtime_version")) return GetRuntimeVersion();
  }
  throw StringError("bad argument");
}
