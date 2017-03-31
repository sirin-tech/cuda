#include "common.h"
#include "erlang_port.h"

int main(void) {
  ErlangPort port;
  // NOTE: add all functions here and in erlang_port.h
  port.AddHandler("info", Info);
  port.AddHandler("init", Init);
  port.AddHandler("compile", Compile);
  port.AddHandler("memory_load", MemoryLoad);
  port.AddHandler("memory_read", MemoryRead);
  port.AddHandler("memory_unload", MemoryUnload);
  port.AddHandler("run", Run);
  try {
    // enter port loop
    port.Loop();
  } catch(...) {} // exit normally on any errors - just die
  return 0;
}
