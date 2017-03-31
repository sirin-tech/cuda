#ifndef __ERLANG_PORT_H__
#define __ERLANG_PORT_H__

#include <iostream>
#include <algorithm>
#include <map>
#include <ext/stdio_filebuf.h>

#include "common.h"
#include "driver.h"

#define PORTIN_FILENO  3
#define PORTOUT_FILENO 4

class ErlangPort;
typedef ETERM *(*ErlangHandler)(ErlangPort *port, ETERM *arg);

class ErlangPort {
private:
  std::istream input;
  std::ostream output;
  std::map<std::string, ErlangHandler> handlers;
  ETERM *tuple = NULL;
  ETERM *func = NULL;
  ETERM *arg = NULL;
  ETERM *result = NULL;

  uint32_t ReadPacketLength();
  void WritePacketLength(uint32_t len);
public:
  Driver *driver = NULL;

  ErlangPort();
  ~ErlangPort();
  void Loop();
  void AddHandler(std::string name, ErlangHandler handler);
  void RemoveHandler(std::string name);
};

// API functions
ETERM *Info(ErlangPort *port, ETERM *arg);
ETERM *Init(ErlangPort *port, ETERM *arg);
ETERM *Compile(ErlangPort *port, ETERM *arg);
ETERM *MemoryLoad(ErlangPort *port, ETERM *arg);
ETERM *MemoryRead(ErlangPort *port, ETERM *arg);
ETERM *MemoryUnload(ErlangPort *port, ETERM *arg);
ETERM *Run(ErlangPort *port, ETERM *arg);

#endif // __ERLANG_PORT_H__
