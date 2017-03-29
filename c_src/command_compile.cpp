#include "common.h"
#include "erlang_port.h"
#include "driver.h"

ETERM *Compile(ErlangPort *port, ETERM *arg) {
  if (!port->driver) throw StringError("Driver not initialized");
  if (!ERL_IS_TUPLE(arg) || erl_size(arg) != 2) throw StringError("Bad argument");

  auto sourcesArg = erl_element(1, arg);
  if (!ERL_IS_LIST(sourcesArg)) throw StringError("Bad argument");
  std::list<std::string> sources;
  auto size = erl_length(sourcesArg);
  if (size < 1) throw StringError("Bad argument");
  for (int i = 0; i < size; i++) {
    auto srcBin = erl_hd(sourcesArg);
    sourcesArg = erl_tl(sourcesArg);
    if (!ERL_IS_BINARY(srcBin)) throw StringError("Bad argument");
    std::string src((char *)ERL_BIN_PTR(srcBin), erl_size(srcBin));
    sources.push_back(src);
  }

  auto optionsArg = erl_element(2, arg);
  struct LinkerOptions options;
  options.maxRegisters = -1;
  options.threadsPerBlock = -1;
  options.optimizationLevel = -1;
  options.target = -1;
  options.debug = -1;
  options.verbose = -1;
  options.infoSize = -1;
  options.errorSize = -1;
  if (!IS_NIL(optionsArg)) {
    auto opts = GetKeywords(optionsArg);
    auto it = opts.find("max_registers");
    if (it != opts.end()) {
      options.maxRegisters = Get<int>(it->second);
      if (options.maxRegisters < 0) throw StringError("Bad argument");
    }
    it = opts.find("threads_per_block");
    if (it != opts.end()) {
      options.threadsPerBlock = Get<int>(it->second);
      if (options.threadsPerBlock < 0) throw StringError("Bad argument");
    }
    it = opts.find("optimization_level");
    if (it != opts.end()) {
      options.optimizationLevel = Get<int>(it->second);
      if (options.optimizationLevel < 0 || options.optimizationLevel > 4) throw StringError("Bad argument");
    }
    // TODO: target parsing here
    // it = opts.find("target");
    it = opts.find("debug");
    if (it != opts.end()) options.debug = Get<bool>(it->second) ? 1 : 0;
    it = opts.find("verbose");
    if (it != opts.end()) options.verbose = Get<bool>(it->second) ? 1 : 0;
  }

  auto module = port->driver->Compile(sources, options);
  return FORMAT("{~a,~i}", OK_STR, module);
}
