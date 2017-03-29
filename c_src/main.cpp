#include "common.h"
#include "erlang_port.h"

int main(void) {
  ErlangPort port;
  // NOTE: add all functions here and in erlang_port.h
  port.AddHandler("info", Info);
  port.AddHandler("init", Init);
  port.AddHandler("compile", Compile);
  try {
    // enter port loop
    port.Loop();
  } catch(...) {} // exit normally on any errors - just die
  return 0;
}
