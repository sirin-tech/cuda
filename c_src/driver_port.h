#ifndef __DRIVER_PORT_H__
#define __DRIVER_PORT_H__

#include "erlang_port.h"

class DriverPort: public ErlangPort {
private:
  Driver *driver = NULL;

  template <typename T> T Unpack(ETERM *term);
protected:
  virtual ETERM *HandleTermFunction(std::string name, ETERM *arg);
  virtual ETERM *HandleRawFunction(std::string name, RawData &data, size_t size);

  std::shared_ptr<RunArguments> UnpackRunArguments(ETERM *term);

  ETERM *Compile(ETERM *arg);
  ETERM *MemoryRead(ETERM *arg);
  ETERM *MemoryUnload(ETERM *arg);
  ETERM *ModuleLoad(ETERM *arg);
  ETERM *MemoryShare(ETERM *arg);
  ETERM *Run(ETERM *arg);
  ETERM *MemoryLoad(RawData &data, size_t size);
  ETERM *MemoryLoad(ETERM *arg);
  ETERM *Stream(ETERM *arg);
  ETERM *DeviceInfo();
public:
  DriverPort(int device);
  ~DriverPort();
};

#endif // __DRIVER_PORT_H__
