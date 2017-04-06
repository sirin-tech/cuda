#include "utils.h"
#include "common.h"
#include "cuda_runtime.h"

int BestDevice() {
  int maxPerfDevice = 0;
  int smPerMultiproc = 0;
  int currentDevice;
  int bestArch = 0;
  int prohibited = 0;
  int deviceCount = 0;
  unsigned long long maxComputePerf = 0;
  cudaError_t result;

  cudaDeviceProp deviceProp;
  result = cudaGetDeviceCount(&deviceCount);
  if (result != cudaSuccess) throw RuntimeError(result);
  if (deviceCount == 0) throw StringError("No devices supporting CUDA");

  // Find the best major SM Architecture GPU device
  for (currentDevice = 0; currentDevice < deviceCount; currentDevice++) {
    result = cudaGetDeviceProperties(&deviceProp, currentDevice);
    if (result != cudaSuccess) throw RuntimeError(result);
    if (deviceProp.computeMode == cudaComputeModeProhibited) {
      prohibited++;
      continue;
    }
    // If this GPU is not running on Compute Mode prohibited, then we can add it to the list
    if (deviceProp.major > 0 && deviceProp.major < 9999) {
      bestArch = std::max(bestArch, deviceProp.major);
    }
  }

  if (prohibited == deviceCount) {
    throw StringError("All devices have compute mode prohibited");
  }

  for (currentDevice = 0; currentDevice < deviceCount; currentDevice++) {
    cudaGetDeviceProperties(&deviceProp, currentDevice);
    if (deviceProp.computeMode == cudaComputeModeProhibited) continue;
    smPerMultiproc = deviceProp.major == 9999 && deviceProp.minor == 9999 ?
                     1 :
                     SMVer2Cores(deviceProp.major, deviceProp.minor);
    unsigned long long computePerf = (unsigned long long) deviceProp.multiProcessorCount * smPerMultiproc * deviceProp.clockRate;
    if (computePerf > maxComputePerf) {
      // If we find GPU with SM major > 2, search only these
      // If our device == bestArch, choose this, or else pass
      if ((bestArch > 2 && deviceProp.major == bestArch) || bestArch <= 2) {
        maxComputePerf = computePerf;
        maxPerfDevice  = currentDevice;
      }
    }
  }
  return maxPerfDevice;
}
