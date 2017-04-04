#include "common.h"
#include "erlang_port.h"

int main(int argc, char *argv[]) {
  int device = -1;
  if (argc > 0) {
    std::string deviceStr(argv[0]);
    try {
      device = std::stoi(deviceStr);
    } catch (const std::invalid_argument &e) {}
  }

  try {
    ErlangPort port(device);

    // NOTE: add all functions here and in erlang_port.h
    port.AddHandler("info", Info);
    // port.AddHandler("init", Init);
    port.AddHandler("compile", Compile);
    port.AddHandler("module_load", ModuleLoad);
    port.AddHandler("memory_read", MemoryRead);
    port.AddHandler("memory_unload", MemoryUnload);
    port.AddHandler("run", Run);
    port.AddRawHandler(MEMORY_LOAD, MemoryLoad);

    // enter port loop
    port.Loop();
  } catch(...) {} // exit normally on any errors - just die
  return 0;
}
