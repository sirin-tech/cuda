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

class ErlangPort;
typedef ETERM *(*ErlangHandler)(ErlangPort *port, ETERM *arg);
typedef ETERM *(*ErlangRawHandler)(ErlangPort *port, std::shared_ptr<void> &data, size_t len);

class ErlangPort {
private:
  std::istream input;
  std::ostream output;
  std::map<std::string, ErlangHandler> handlers;
  std::map<int, ErlangRawHandler> rawHandlers;
  ETERM *tuple = NULL;
  ETERM *func = NULL;
  ETERM *arg = NULL;
  ETERM *result = NULL;

  uint32_t ReadPacketLength();
  uint8_t ReadPacketType();
  uint8_t ReadRawFunc();
  ETERM *ReadPacket(uint32_t len);
  void WritePacketLength(uint32_t len);
  void WritePacket(ETERM *packet);
public:
  Driver *driver = NULL;

  ErlangPort(int device);
  ~ErlangPort();
  void Loop();
  void SendRawReply(void *data, size_t size);
  void AddHandler(std::string name, ErlangHandler handler);
  void AddRawHandler(int id, ErlangRawHandler handler);
  void RemoveHandler(std::string name);
  void RemoveRawHandler(int id);
};

void Init(ErlangPort *port, int device);

// API functions
ETERM *Info(ErlangPort *port, ETERM *arg);
ETERM *Compile(ErlangPort *port, ETERM *arg);
ETERM *MemoryRead(ErlangPort *port, ETERM *arg);
ETERM *MemoryUnload(ErlangPort *port, ETERM *arg);
ETERM *Run(ErlangPort *port, ETERM *arg);

// raw API function
ETERM *MemoryLoad(ErlangPort *port, std::shared_ptr<void> &data, size_t size);

#endif // __ERLANG_PORT_H__
