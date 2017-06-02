#ifndef __ERLANG_PORT_H__
#define __ERLANG_PORT_H__

#include <iostream>
#include <algorithm>
#include <map>
#include <memory>
#include <ext/stdio_filebuf.h>

#include "common.h"
#include "driver.h"

#define PORTIN_FILENO  3
#define PORTOUT_FILENO 4

#define MEMORY_LOAD 1

typedef std::shared_ptr<void> RawData;

class ErlangPort {
private:
  std::istream input;
  std::ostream output;
  ETERM *tuple = NULL;
  ETERM *funcAtom = NULL;
  ETERM *arg = NULL;
  ETERM *result = NULL;

  uint32_t ReadPacketLength();
  uint8_t ReadPacketType();
  uint8_t ReadRawFunc();
  ETERM *ReadTermPacket(uint32_t len);
  void WritePacketLength(uint32_t len);

protected:
  virtual ETERM *HandleTermFunction(std::string name, ETERM *arg) = 0;
  virtual ETERM *HandleRawFunction(std::string name, RawData &data, size_t size) = 0;

public:
  ErlangPort();
  ~ErlangPort();
  void WriteTermPacket(ETERM *packet);
  void WriteRawPacket(void *data, size_t size);
  void Loop();
};

// API functions
ETERM *Info(ErlangPort *port, ETERM *arg);
ETERM *Compile(ErlangPort *port, ETERM *arg);
ETERM *ModuleLoad(ErlangPort *port, ETERM *arg);
ETERM *MemoryRead(ErlangPort *port, ETERM *arg);
ETERM *MemoryUnload(ErlangPort *port, ETERM *arg);
ETERM *Run(ErlangPort *port, ETERM *arg);

// raw API function
ETERM *MemoryLoad(ErlangPort *port, std::shared_ptr<void> &data, size_t size);

#endif // __ERLANG_PORT_H__
