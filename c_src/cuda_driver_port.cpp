#include "common.h"
#include "driver_port.h"

void CleanupCuda() {
  DEBUG("Main function exitted. Resetting CUDA device");
  cudaDeviceReset();
}

int main(int argc, char *argv[]) {
  std::atexit(CleanupCuda);

  int device = -1;
  if (argc > 0) {
    std::string deviceStr(argv[0]);
    try {
      device = std::stoi(deviceStr);
    } catch (const std::invalid_argument &e) {}
  }

  try {
    DriverPort port(device);
    port.Loop();
  } catch(...) {} // exit normally on any errors - just die
  return 0;
}
