#include "driver_port.h"
#include "utils.h"
#include "driver.h"

template <> LinkerOptions DriverPort::Unpack<LinkerOptions>(ETERM *term) {
  LinkerOptions options;
  options.maxRegisters = -1;
  options.threadsPerBlock = -1;
  options.optimizationLevel = -1;
  options.target = -1;
  options.debug = -1;
  options.verbose = -1;
  options.infoSize = -1;
  options.errorSize = -1;
  if (!IS_NIL(term)) {
    auto opts = GetKeywords(term);
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
      if (options.optimizationLevel < 0 || options.optimizationLevel > 4) {
        throw StringError("Bad argument");
      }
    }
    // TODO: target parsing here
    // it = opts.find("target");
    it = opts.find("debug");
    if (it != opts.end()) options.debug = Get<bool>(it->second) ? 1 : 0;
    it = opts.find("verbose");
    if (it != opts.end()) options.verbose = Get<bool>(it->second) ? 1 : 0;
  }
  return options;
}

DriverPort::DriverPort(int device): ErlangPort() {
  try {
    auto result = cuInit(0);
    if (result != CUDA_SUCCESS) throw DriverError(result);
    if (device < 0) device = BestDevice();
    if (device < 0) throw StringError("Where are no GPU devices to initialize");
    driver = new Driver(device);
  } catch (Error &error) {
    WriteTermPacket(error.AsTerm());
    throw StringError("Initializing error");
  }
  DEBUG("DriverPort initialized");
}

DriverPort::~DriverPort() {
  DEBUG("DriverPort destroyed");
  if (driver) delete driver;
}

ETERM *DriverPort::HandleTermFunction(std::string name, ETERM *arg) {
  if (name == "compile") return Compile(arg);
  if (name == "memory_read") return MemoryRead(arg);
  if (name == "memory_unload") return MemoryUnload(arg);
  if (name == "module_load") return ModuleLoad(arg);
  if (name == "run") return Run(arg);
  if (name == "stream") return Stream(arg);
  return NULL;
}

ETERM *DriverPort::HandleRawFunction(std::string name, RawData &data, size_t size) {
  if (name == "memory_load") return MemoryLoad(data, size);
  return NULL;
}

ETERM *DriverPort::Compile(ETERM *arg) {
  DEBUG("Enter DriverPort::Compile");
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

  auto options = Unpack<LinkerOptions>(erl_element(2, arg));
  auto module = driver->Compile(sources, options);
  return FORMAT("{~a,~w}", OK_STR, driver->PackModule(module));
}

ETERM *DriverPort::MemoryRead(ETERM *arg) {
  DEBUG("Enter DriverPort::MemoryRead");
  auto n = GetMemoryIndex(arg);
  auto size = driver->GetMemorySize(n);
  if (size < 0) throw StringError("Invalid memory handle");
  char *data = NULL;
  cuMemAllocHost((void **)&data, size);
  driver->ReadMemory(n, (void *)data);
  WriteRawPacket((void *)data, size);
  return NULL;
}

ETERM *DriverPort::MemoryUnload(ETERM *arg) {
  DEBUG("Enter DriverPort::MemoryUnload");
  auto n = GetMemoryIndex(arg);
  driver->UnloadMemory(n);
  return erl_mk_atom(OK_STR);
}

ETERM *DriverPort::ModuleLoad(ETERM *arg) {
  DEBUG("Enter DriverPort::ModuleLoad");
  if (!ERL_IS_TUPLE(arg) || erl_size(arg) != 2) throw StringError("Bad argument");

  auto srcArg = erl_element(1, arg);
  if (!ERL_IS_BINARY(srcArg)) throw StringError("Bad argument");
  std::string src((char *)ERL_BIN_PTR(srcArg), erl_size(srcArg));

  auto options = Unpack<LinkerOptions>(erl_element(2, arg));
  auto module = driver->LoadModule(src, options);
  return FORMAT("{~a,~w}", OK_STR, driver->PackModule(module));
}

ETERM *DriverPort::Run(ETERM *arg) {
  DEBUG("Enter DriverPort::Run");
  if (!ERL_IS_TUPLE(arg)) throw StringError("Bad argument");
  auto argc = erl_size(arg);
  if (argc < 2) throw StringError("Bad argument");
  auto moduleTerm = erl_element(1, arg);
  auto funcTerm   = erl_element(2, arg);
  if (!ERL_IS_BINARY(funcTerm)) throw StringError("Bad argument");

  auto module = GetModuleIndex(moduleTerm);
  std::string func((char *)ERL_BIN_PTR(funcTerm), erl_size(funcTerm));
  int gx = 1, gy = 1, gz = 1;
  int bx = 1, by = 1, bz = 1;
  std::shared_ptr<RunArguments> argsPtr;

  if (argc > 2) {
    ETERM *grid = NULL;
    ETERM *block = NULL;
    ETERM *params = NULL;

    if (argc == 3) {
      params = erl_element(3, arg);
      if (ERL_IS_TUPLE(params) || ERL_IS_INTEGER(params)) {
        block = params;
        params = NULL;
      }
    } else if (argc == 4) {
      block = erl_element(3, arg);
      params = erl_element(4, arg);
      if (ERL_IS_TUPLE(params) || ERL_IS_INTEGER(params)) {
        grid = params;
        params = NULL;
      }
    } else if (argc == 5) {
      block  = erl_element(3, arg);
      grid   = erl_element(4, arg);
      params = erl_element(5, arg);
    } else {
      throw StringError("Bad argument");
    }

    if (block) {
      if (ERL_IS_INTEGER(block)) {
        bx = Get<int>(block);
      } else if (ERL_IS_TUPLE(block)) {
        auto s = erl_size(block);
        if (s > 0) bx = Get<int>(erl_element(1, block));
        if (s > 1) by = Get<int>(erl_element(2, block));
        if (s > 2) bz = Get<int>(erl_element(3, block));
      } else {
        throw StringError("Bad argument");
      }
    }
    if (grid) {
      if (ERL_IS_INTEGER(grid)) {
        bx = Get<int>(grid);
      } else if (ERL_IS_TUPLE(grid)) {
        auto s = erl_size(grid);
        if (s > 0) gx = Get<int>(erl_element(1, grid));
        if (s > 1) gy = Get<int>(erl_element(2, grid));
        if (s > 2) gz = Get<int>(erl_element(3, grid));
      } else {
        throw StringError("Bad argument");
      }
    }
    argsPtr = params ? UnpackRunArguments(params) : NULL;
  }

  auto params = std::make_tuple(func, gx, gy, gz, bx, by, bz);
  driver->Run(module, params, argsPtr);
  DEBUG("Leave DriverPort::Run");
  return erl_mk_atom(OK_STR);
}

ETERM *DriverPort::Stream(ETERM *arg) {
  DEBUG("Enter DriverPort::Stream");
  if (!ERL_IS_TUPLE(arg)) throw StringError("Bad argument");
  auto argc = erl_size(arg);
  if (argc < 2) throw StringError("Bad argument");
  auto moduleTerm = erl_element(1, arg);
  auto batchTerm  = erl_element(2, arg);
  if (!ERL_IS_LIST(batchTerm)) throw StringError("Bad argument");

  auto module = GetModuleIndex(moduleTerm);
  std::vector<RunEnvironment> batch;

  auto s = erl_length(batchTerm);
  for (int i = 0; i < s; i++) {
    auto item = erl_hd(batchTerm);
    batchTerm = erl_tl(batchTerm);

    if (!ERL_IS_TUPLE(item)) throw StringError("Bad argument");
    if (erl_size(item) != 4) throw StringError("Bad argument");
    auto funcTerm = erl_element(1, item);
    auto bTerm    = erl_element(2, item);
    auto gTerm    = erl_element(3, item);
    auto argsTerm = erl_element(4, item);

    std::string func((char *)ERL_BIN_PTR(funcTerm), erl_size(funcTerm));

    if (!ERL_IS_TUPLE(bTerm)) throw StringError("Bad argument");
    if (erl_size(bTerm) != 3) throw StringError("Bad argument");
    auto bx = ERL_INT_VALUE(erl_element(1, bTerm));
    auto by = ERL_INT_VALUE(erl_element(2, bTerm));
    auto bz = ERL_INT_VALUE(erl_element(3, bTerm));

    if (!ERL_IS_TUPLE(gTerm)) throw StringError("Bad argument");
    if (erl_size(gTerm) != 3) throw StringError("Bad argument");
    auto gx = ERL_INT_VALUE(erl_element(1, gTerm));
    auto gy = ERL_INT_VALUE(erl_element(2, gTerm));
    auto gz = ERL_INT_VALUE(erl_element(3, gTerm));

    auto args = UnpackRunArguments(argsTerm);
    auto params = std::make_tuple(func, gx, gy, gz, bx, by, bz);
    batch.push_back(std::make_tuple(params, args));
  }

  driver->Stream(module, batch);
  return erl_mk_atom(OK_STR);
}

ETERM *DriverPort::MemoryLoad(RawData &data, size_t size) {
  DEBUG("Enter DriverPort::MemoryLoad");
  int n = driver->LoadMemory(data.get(), size);
  return FORMAT("{~a,~w}", OK_STR, driver->PackMemory(n));
}

std::shared_ptr<RunArguments> DriverPort::UnpackRunArguments(ETERM *term) {
  std::shared_ptr<RunArguments> args = std::make_shared<RunArguments>(RunArguments());

  if (!ERL_IS_LIST(term)) throw StringError("Bad argument");
  auto s = erl_length(term);
  for (int i = 0; i < s; i++) {
    auto param = erl_hd(term);
    term = erl_tl(term);
    if (ERL_IS_TUPLE(param)) {
      auto param_type = erl_element(1, param);
      auto param_value = erl_element(2, param);
      if (ERL_IS_ATOM(param_type) && ATOM_EQ(param_type, "memory")) {
        auto mem = driver->GetMemory(Get<int>(param_value));
        if (!mem) throw StringError("Invalid memory handle");
        args->Add(*mem);
      }
    } else if (ERL_IS_INTEGER(param)) {
      args->Add(ERL_INT_VALUE(param));
    } else if (ERL_IS_FLOAT(param)) {
      float f = ERL_FLOAT_VALUE(param);
      args->Add(f);
    } else {
      throw StringError("Bad argument");
    }
  }
  return args;
}
