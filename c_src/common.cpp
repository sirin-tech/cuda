#include "common.h"

ETERM *CudaRuntimeError(cudaError_t error) {
  return ERROR(cudaGetErrorString(error));
}

ETERM *CudaDriverError(CUresult error) {
  const char *buf;
  if (cuGetErrorString(error, &buf) == CUDA_SUCCESS) return ERROR(buf);
  return ERROR("Unknown error");
}


/* ETERM *CudaError(cudaError_t error) {
  switch (error) {
    case cudaSuccess: return erl_format("~a", "ok");
    case cudaErrorMissingConfiguration: return ERROR("cudaConfigureCall() expected");
    case cudaErrorMemoryAllocation: return ERROR("Unable to allocate enough memory");
    case cudaErrorInitializationError: return ERROR("CUDA not initialized");
    case cudaErrorLaunchFailure: return ERROR("Error executing kernel");
    case cudaErrorPriorLaunchFailure: return ERROR("Previous kernel launch failed");
    case cudaErrorLaunchTimeout: return ERROR("Kernel execution timeout");
    case cudaErrorLaunchOutOfResources: return ERROR("Out of resources");
    case cudaErrorInvalidDeviceFunction: return ERROR("Invalid device function");
    case cudaErrorInvalidConfiguration: return ERROR("Invalid configuration");
    case cudaErrorInvalidDevice: return ERROR("Invalid device");
    case cudaErrorInvalidValue: return ERROR("Invalid value");
    case cudaErrorInvalidPitchValue: return ERROR("Invalid putch value");
    case cudaErrorInvalidSymbol: return ERROR("Invalid symbol");
    case cudaErrorMapBufferObjectFailed: return ERROR("Buffer object could not be mapped");
    case cudaErrorUnmapBufferObjectFailed: return ERROR("Buffer object could not be unmapped");
    case cudaErrorInvalidHostPointer: return ERROR("Invalid host pointer");
    case cudaErrorInvalidDevicePointer: return ERROR("Invalid device pointer");
    case cudaErrorInvalidTexture: return ERROR("Invalid texture");
    case cudaErrorInvalidTextureBinding: return ERROR("Invalid texture binding");
    case cudaErrorInvalidChannelDescriptor: return ERROR("Invalid channel descriptor");
    case cudaErrorInvalidMemcpyDirection: return ERROR("Wrong memcpy direction");
    case cudaErrorAddressOfConstant: return ERROR("Addresses of constants forbidden until CUDA 3.1");
    case cudaErrorTextureFetchFailed: return ERROR("Texture ftech failed");
    case cudaErrorTextureNotBound: return ERROR("Texture not bound");
    case cudaErrorSynchronizationError: return ERROR("Syncronization error");
    case cudaErrorInvalidFilterSetting: return ERROR("Invalid filter settings");
    case cudaErrorInvalidNormSetting: return ERROR("Invalid normalized setting");
    case cudaErrorMixedDeviceExecution: return ERROR("Mixed device execution");
    case cudaErrorCudartUnloading: return ERROR("CUDA runtime unloading");
    case cudaErrorUnknown: return ERROR("Unknown error");
    case cudaErrorNotYetImplemented: return ERROR("Not yet implemented");
    case cudaErrorMemoryValueTooLarge: return ERROR("Memory value too large");
    case cudaErrorInvalidResourceHandle: return ERROR("Invalid resource handle");
    case cudaErrorNotReady: return ERROR("Asynchronous operation not ready");
    case cudaErrorInsufficientDriver: return ERROR("Insufficient driver");
    case cudaErrorSetOnActiveProcess: return ERROR("Set on active process");
    case cudaErrorInvalidSurface: return ERROR("Invalid surface");
    case cudaErrorNoDevice: return ERROR("No device");
    case cudaErrorECCUncorrectable: return ERROR("Uncorrectable ECC error");
    case cudaErrorSharedObjectSymbolNotFound: return ERROR("Shared object symbol not found");
    case cudaErrorSharedObjectInitFailed: return ERROR("Shared object init failed");
    case cudaErrorUnsupportedLimit: return ERROR("Unsupported limit");
    // TODO: Finish error list. After finishing remove default section
    // http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e00383e8aef5398ee38e28ed41e357b48917c
    default: return ERROR("Unimplemented error");
    // case : return ERROR("");
    // case : return ERROR("");
    // case : return ERROR("");
    // case : return ERROR("");
    // case : return ERROR("");
    // case : return ERROR("");
    // case : return ERROR("");
    // case : return ERROR("");
    // case : return ERROR("");
    // case : return ERROR("");
    // case : return ERROR("");
    // case : return ERROR("");
    // case : return ERROR("");
    // case : return ERROR("");
  }
} */
