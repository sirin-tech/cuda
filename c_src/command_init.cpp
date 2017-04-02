#include "erlang_port.h"

inline int SMVer2Cores(int major, int minor) {
  // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
  typedef struct {
    int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
    int Cores;
  } SM2Cores;

  SM2Cores coresPerSM[] = {
    {0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
    {0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
    {0x30, 192}, // Kepler Generation (SM 3.0) GK10x class
    {0x32, 192}, // Kepler Generation (SM 3.2) GK10x class
    {0x35, 192}, // Kepler Generation (SM 3.5) GK11x class
    {0x37, 192}, // Kepler Generation (SM 3.7) GK21x class
    {0x50, 128}, // Maxwell Generation (SM 5.0) GM10x class
    {0x52, 128}, // Maxwell Generation (SM 5.2) GM20x class
    {0x53, 128}, // Maxwell Generation (SM 5.3) GM20x class
    {0x60, 64 }, // Pascal Generation (SM 6.0) GP100 class
    {0x61, 128}, // Pascal Generation (SM 6.1) GP10x class
    {0x62, 128}, // Pascal Generation (SM 6.2) GP10x class
    {  -1, -1 }
  };

  int idx = 0;
  for(idx = 0; coresPerSM[idx].SM != -1; idx++) {
    if (coresPerSM[idx].SM == ((major << 4) + minor)) {
      return coresPerSM[idx].Cores;
    }
  }
  return coresPerSM[idx - 1].Cores;
}

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

void Init(ErlangPort *port, int device = -1) {
  if (port->driver) throw StringError("Already initialized");
  auto result = cuInit(0);
  if (result != CUDA_SUCCESS) throw DriverError(result);
  // int device = -1;
  if (device < 0) device = BestDevice();
  // if (IS_NIL(arg)) {
  //   device = BestDevice();
  // } else if (ERL_IS_INTEGER(arg)) {
  //   device = ERL_INT_VALUE(arg);
  // }
  if (device < 0) throw StringError("Where are no GPU devices to initialize");
  port->driver = new Driver(device);
  // return erl_mk_atom(OK_STR);
}
