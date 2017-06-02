#include "driver_port.h"
#include "utils.h"
#include "driver.h"
#include "commands.h"

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
    if (result != CUDA_SUCCESS) throw DriverError(result, "DriverPort initialize");
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
  if (name == "memory_share") return MemoryShare(arg);
  if (name == "memory_load") return MemoryLoad(arg);
  if (name == "module_load") return ModuleLoad(arg);
  if (name == "run") return Run(arg);
  if (name == "stream") return Stream(arg);
  if (name == "device_info") return DeviceInfo();
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
  return FORMAT("{ok,~w}", driver->PackModule(module));
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

ETERM *DriverPort::MemoryShare(ETERM *arg) {
  DEBUG("Enter DriverPort::MemoryShare");
  auto n = GetMemoryIndex(arg);
  auto mem = driver->ShareMemory(n);
  return FORMAT("{ok,~w}", driver->PackMemory(mem));
}

ETERM *DriverPort::MemoryLoad(ETERM *arg) {
  DEBUG("Enter DriverPort::MemoryLoad");
  auto mem = driver->Unpack<SharedMemory>(arg);
  int n = driver->LoadMemory(mem);
  return FORMAT("{ok,~w}", driver->PackMemory(n));
}

ETERM *DriverPort::ModuleLoad(ETERM *arg) {
  DEBUG("Enter DriverPort::ModuleLoad");
  if (!ERL_IS_TUPLE(arg) || erl_size(arg) != 2) throw StringError("Bad argument");

  auto srcArg = erl_element(1, arg);
  if (!ERL_IS_BINARY(srcArg)) throw StringError("Bad argument");
  std::string src((char *)ERL_BIN_PTR(srcArg), erl_size(srcArg));

  auto options = Unpack<LinkerOptions>(erl_element(2, arg));
  auto module = driver->LoadModule(src, options);
  return FORMAT("{ok,~w}", driver->PackModule(module));
}

ETERM *DriverPort::Run(ETERM *arg) {
  DEBUG("Enter DriverPort::Run");
  if (!ERL_IS_TUPLE(arg)) throw StringError("Bad argument");
  auto argc = erl_size(arg);
  if (argc < 2) throw StringError("Bad argument");
  auto moduleTerm  = erl_element(1, arg);
  auto commandTerm = erl_element(2, arg);

  auto module = GetModuleIndex(moduleTerm);
  Commands::Context ctx;
  ctx.module = driver->GetModule(module);
  auto cmd = Commands::Command::Create(driver, commandTerm);
  cmd->Run(ctx);
  DEBUG("Leave DriverPort::Run");
  return erl_mk_atom(OK_STR);
}

ETERM *DriverPort::Stream(ETERM *arg) {
  DEBUG("Enter DriverPort::Stream");
  if (!ERL_IS_TUPLE(arg)) throw StringError("Bad argument");
  auto argc = erl_size(arg);
  if (argc < 2) throw StringError("Bad argument");
  auto moduleTerm  = erl_element(1, arg);
  auto commandTerm = erl_element(2, arg);
  if (!ERL_IS_LIST(commandTerm)) throw StringError("Bad argument");

  auto module = GetModuleIndex(moduleTerm);
  Commands::Context ctx;
  ctx.module = driver->GetModule(module);
  auto cmd = Commands::Command::Create(driver, commandTerm);
  cmd->Run(ctx);

  return erl_mk_atom(OK_STR);
}

ETERM *DriverPort::MemoryLoad(RawData &data, size_t size) {
  DEBUG("Enter raw DriverPort::MemoryLoad");
  int n = driver->LoadMemory(data.get(), size);
  return FORMAT("{ok,~w}", driver->PackMemory(n));
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

ETERM *DriverPort::DeviceInfo() {
  int v[44];
  CUdevice_attribute c[44] = {
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
    CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY,
    CU_DEVICE_ATTRIBUTE_WARP_SIZE,
    CU_DEVICE_ATTRIBUTE_MAX_PITCH,
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK,
    CU_DEVICE_ATTRIBUTE_CLOCK_RATE,
    CU_DEVICE_ATTRIBUTE_GPU_OVERLAP,
    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
    CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT,
    CU_DEVICE_ATTRIBUTE_INTEGRATED,
    CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY,
    CU_DEVICE_ATTRIBUTE_COMPUTE_MODE,
    CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS,
    CU_DEVICE_ATTRIBUTE_ECC_ENABLED,
    CU_DEVICE_ATTRIBUTE_PCI_BUS_ID,
    CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID,
    CU_DEVICE_ATTRIBUTE_TCC_DRIVER,
    CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE,
    CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH,
    CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE,
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
    CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
    CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED,
    CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR,
    CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY,
    CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD,
    CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID,
    CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED,
    CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO,
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS,
    CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS,
    CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED,
    CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM,
  };
  const char *b[18];
  CUresult x;
  CUdevice d = driver->GetHandle();
  for (int i = 0; i < 44; i++) {
    x = cuDeviceGetAttribute(&v[i], c[i], d);
    if (x != CUDA_SUCCESS) throw DriverError(x);
  }
  b[0] = v[13] == 1 ? "true" : "false";
  b[1] = v[15] == 1 ? "true" : "false";
  b[2] = v[16] == 1 ? "true" : "false";
  b[3] = v[17] == 1 ? "true" : "false";
  b[4] = "unknown";
  switch (v[18]) {
    case CU_COMPUTEMODE_DEFAULT: b[4] = "default"; break;
    case CU_COMPUTEMODE_PROHIBITED: b[4] = "prohibited"; break;
    case CU_COMPUTEMODE_EXCLUSIVE_PROCESS: b[4] = "exclusive_process"; break;
  }
  b[5] = v[19] == 1 ? "true" : "false";
  b[6] = v[20] == 1 ? "true" : "false";
  b[7] = v[23] == 1 ? "true" : "false";
  b[8] = v[28] == 1 ? "true" : "false";
  b[9] = v[31] == 1 ? "true" : "false";
  b[10] = v[32] == 1 ? "true" : "false";
  b[11] = v[35] == 1 ? "true" : "false";
  b[12] = v[36] == 1 ? "true" : "false";
  b[13] = v[38] == 1 ? "true" : "false";
  b[14] = v[40] == 1 ? "true" : "false";
  b[15] = v[41] == 1 ? "true" : "false";
  b[16] = v[42] == 1 ? "true" : "false";
  b[17] = v[43] == 1 ? "true" : "false";
  return FORMAT(
    "{ok,["
      "{max_threads_per_block,~i},"
      "{max_block,{~i,~i,~i}},"
      "{max_grid,{~i,~i,~i}},"
      "{max_shared_memory_per_block,~i},"
      "{total_constant_memory,~i},"
      "{warp_size,~i},"
      "{max_pitch,~i},"
      "{max_registers_per_block,~i},"
      "{clock_rate,~i},"
      "{gpu_overlap,~a},"
      "{miltiprocessor_count,~i},"
      "{kernel_exec_timeout,~a},"
      "{integrated,~a},"
      "{can_map_host_memory,~a},"
      "{compute_mode,~a},"
      "{concurrent_kernels,~a},"
      "{ecc_enabled,~a},"
      "{pci_bus_id,~i},"
      "{pci_device_id,~i},"
      "{tcc_driver,~a},"
      "{memory_clock_rate,~i},"
      "{global_memory_bus_width,~i},"
      "{l2_cache_size,~i},"
      "{max_threads_per_multiprocessor,~i},"
      "{unified_arressing,~a},"
      "{compute_capability,{~i,~i}},"
      "{global_l1_cache_supported,~a},"
      "{glocal_l1_cache_supported,~a},"
      "{max_shared_memory_per_multiprocessor,~i},"
      "{max_registers_per_multiprocessor,~i},"
      "{managed_memory,~a},"
      "{multi_gpu_board,~a},"
      "{multi_gpu_board_group_id,~i},"
      "{host_native_atomic_supported,~a},"
      "{single_to_double_precision_perf_ratio,~i},"
      "{pageable_memory_access,~a},"
      "{concurrent_managed_access,~a},"
      "{compute_preemption_supported,~a},"
      "{can_use_host_pointer_for_registered_mem,~a}"
    "]}",
    v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11],
    v[12], b[0], v[14], b[1], b[2], b[3], b[4], b[5], b[6], v[21], v[22], b[7],
    v[24], v[25], v[26], v[27], b[8], v[29], v[30], b[9], b[10], v[33], v[34],
    b[11], b[12], v[37], b[13], v[39], b[14], b[15], b[16], b[17]);
}
