#ifndef __RUNTIME_PORT_H__
#define __RUNTIME_PORT_H__

#include "erlang_port.h"

class RuntimePort: public ErlangPort {
protected:
  virtual ETERM *HandleTermFunction(std::string name, ETERM *arg);
  virtual ETERM *HandleRawFunction(std::string name, RawData &data, size_t size);

  ETERM *Info(ETERM *arg);
public:
  RuntimePort(int device);
  ~RuntimePort();
};

#endif // __RUNTIME_PORT_H__
