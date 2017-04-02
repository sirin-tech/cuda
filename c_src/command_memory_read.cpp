#include "common.h"
#include "driver.h"
#include "erlang_port.h"

ETERM *MemoryRead(ErlangPort *port, ETERM *arg) {
  if (!port->driver) throw StringError("Driver not initialized");
  auto n = GetMemoryIndex(arg);
  auto size = port->driver->GetMemorySize(n);
  if (size < 0) throw StringError("Invalid memory handle");
  // char *data = (char *)malloc(size);
  char *data = NULL;
  cuMemAllocHost((void **)&data, size);
  port->driver->ReadMemory(n, (void *)data);
  port->SendRawReply((void *)data, size);
  // return FORMAT("{~a,~w}", OK_STR, erl_mk_binary(data, size));
  return NULL;
}
