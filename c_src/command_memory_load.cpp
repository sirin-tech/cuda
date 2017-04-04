#include "driver.h"
#include "erlang_port.h"

ETERM *MemoryLoad(ErlangPort *port, std::shared_ptr<void> &data, size_t size) {
  if (!port->driver) throw StringError("Driver not initialized");
  int n = port->driver->LoadMemory(data.get(), size);
  return FORMAT("{~a,~w}", OK_STR, port->driver->PackMemory(n));
}
