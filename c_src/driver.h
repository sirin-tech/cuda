#ifndef __DRIVER_H__
#define __DRIVER_H__

#include "common.h"

class Driver {
private:
  CUdevice    device;
  CUcontext   context;
public:
  Driver(int deviceNo);
  ~Driver();
  // void Execute(std::list<std::string> sources);
};

#endif // __DRIVER_H__
