#include "driver.h"
#include "erlang_port.h"

/* ETERM *MemoryLoad(ErlangPort *port, ETERM *arg) {
  if (!port->driver) throw StringError("Driver not initialized");
  if (ERL_IS_BINARY(arg)) {
    size_t size = erl_size(arg);
    int n = port->driver->LoadMemory(ERL_BIN_PTR(arg), size);
    return FORMAT("{~a,~w}", OK_STR, port->driver->PackMemory(n));
  } else {
    throw StringError("Not implemented yet");
  }
} */

ETERM *MemoryLoad(ErlangPort *port, std::shared_ptr<void> &data, size_t size) {
  if (!port->driver) throw StringError("Driver not initialized");
  int n = port->driver->LoadMemory(data.get(), size);
  return FORMAT("{~a,~w}", OK_STR, port->driver->PackMemory(n));
}
