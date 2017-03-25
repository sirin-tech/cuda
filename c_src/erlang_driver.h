#ifndef __ERLANG_DRIVER_H__
#define __ERLANG_DRIVER_H__

#include <iostream>
#include <algorithm>
#include <map>
#include <cstring>
#include <stdio.h>
#include <ext/stdio_filebuf.h>

extern "C" {
  #include "erl_interface.h"
  #include "ei.h"
}

#define PORTIN_FILENO  3
#define PORTOUT_FILENO 4

typedef ETERM *(*ErlangHandler)(ETERM *arg);

class ErlangDriver {
private:
  std::istream input;
  std::ostream output;
  std::map<std::string, ErlangHandler> handlers;
  ETERM *tuple;
  ETERM *func;
  ETERM *arg;
  ETERM *result;

  uint32_t ReadPacketLength();
  void WritePacketLength(uint32_t len);
public:
  ErlangDriver();
  ~ErlangDriver();
  void Loop();
  void AddHandler(std::string name, ErlangHandler handler);
  void RemoveHandler(std::string name);
};

#endif // __ERLANG_DRIVER_H__
