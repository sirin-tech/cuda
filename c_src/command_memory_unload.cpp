#include "driver.h"
#include "erlang_port.h"

ETERM *MemoryUnload(ErlangPort *port, ETERM *arg) {
  if (!port->driver) throw StringError("Driver not initialized");
  auto n = Get<int>(arg);
  port->driver->UnloadMemory(n);
  return erl_mk_atom(OK_STR);
}
