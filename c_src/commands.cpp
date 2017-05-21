#include "commands.h"

namespace Commands {

Command *Command::Create(Driver *driver, ETERM *item) {
  if (ERL_IS_LIST(item)) {
    if (erl_length(item) == 0) return new Batch(driver, item);
    if (ERL_IS_LIST(erl_hd(item))) return new BatchList(driver, item);
    return new Batch(driver, item);
  }
  if (!ERL_IS_TUPLE(item)) throw StringError("Bad argument");
  if (erl_size(item) != 2) throw StringError("Bad argument");
  auto cmd  = erl_element(1, item);
  auto args = erl_element(2, item);
  if (!ERL_IS_ATOM(cmd)) throw StringError("Bad argument");
  if (ATOM_EQ(cmd, "run")) {
    return new RunCommand(driver, args);
  } else {
    throw StringError("Bad command");
  }
}

Batch::Batch(Driver *driver, ETERM *args) : Command(driver) {
  if (!ERL_IS_LIST(args)) throw StringError("Bad argument");
  auto bs = erl_length(args);
  for (int j = 0; j < bs; j++) {
    commands.push_back(Command::Create(driver, erl_hd(args)));
    args = erl_tl(args);
  }
}

void Batch::Run(Context &ctx) {
  CUresult result;

  result = cuStreamCreate(&ctx.stream, CU_STREAM_NON_BLOCKING);
  if (result != CUDA_SUCCESS) throw DriverError(result, "Driver:stream_create");

  for (auto it = commands.begin(); it != commands.end(); ++it) {
    Command *cmd = *it;
    cmd->Run(ctx);
  }
  // DEBUG("Wait DriverPort::Stream");
  result = cuStreamSynchronize(ctx.stream);
  if (result != CUDA_SUCCESS) throw DriverError(result, "Driver:stream_wait");
  result = cuStreamDestroy(ctx.stream);
  if (result != CUDA_SUCCESS) throw DriverError(result, "Driver:stream_free");
}

BatchList::BatchList(Driver *driver, ETERM *args) : Command(driver) {
  if (!ERL_IS_LIST(args)) throw StringError("Bad argument");
  auto bs = erl_length(args);
  for (int j = 0; j < bs; j++) {
    batches.push_back(Command::Create(driver, erl_hd(args)));
    args = erl_tl(args);
  }
}

void BatchList::Run(Context &ctx) {
  for (auto it = batches.begin(); it != batches.end(); ++it) {
    Command *cmd = *it;
    Context batchCtx = ctx;
    cmd->Run(batchCtx);
  }
}

RunCommand::RunCommand(Driver *driver, ETERM *args) : Command(driver) {
  if (!ERL_IS_TUPLE(args)) throw StringError("Bad argument");
  auto argc = erl_size(args);
  if (argc < 1) throw StringError("Bad argument");

  auto kernelTerm = erl_element(1, args);
  if (!ERL_IS_BINARY(kernelTerm)) throw StringError("Bad argument");
  kernel = std::string((char *)ERL_BIN_PTR(kernelTerm), erl_size(kernelTerm));

  gx = 1, gy = 1, gz = 1;
  bx = 1, by = 1, bz = 1;
  arguments = std::make_shared<RunArguments>(RunArguments());

  if (argc > 1) {
    ETERM *grid = NULL;
    ETERM *block = NULL;
    ETERM *params = NULL;

    if (argc == 2) {
      params = erl_element(2, args);
      if (ERL_IS_TUPLE(params) || ERL_IS_INTEGER(params)) {
        block = params;
        params = NULL;
      }
    } else if (argc == 3) {
      block  = erl_element(2, args);
      params = erl_element(3, args);
      if (ERL_IS_TUPLE(params) || ERL_IS_INTEGER(params)) {
        grid = params;
        params = NULL;
      }
    } else if (argc == 4) {
      block  = erl_element(2, args);
      grid   = erl_element(3, args);
      params = erl_element(4, args);
    } else {
      throw StringError("Bad argument");
    }

    if (block) {
      if (ERL_IS_INTEGER(block)) {
        bx = Get<unsigned int>(block);
      } else if (ERL_IS_TUPLE(block)) {
        auto s = erl_size(block);
        if (s > 0) bx = Get<unsigned int>(erl_element(1, block));
        if (s > 1) by = Get<unsigned int>(erl_element(2, block));
        if (s > 2) bz = Get<unsigned int>(erl_element(3, block));
      } else {
        throw StringError("Bad argument");
      }
    }

    if (grid) {
      if (ERL_IS_INTEGER(grid)) {
        bx = Get<unsigned int>(grid);
      } else if (ERL_IS_TUPLE(grid)) {
        auto s = erl_size(grid);
        if (s > 0) gx = Get<unsigned int>(erl_element(1, grid));
        if (s > 1) gy = Get<unsigned int>(erl_element(2, grid));
        if (s > 2) gz = Get<unsigned int>(erl_element(3, grid));
      } else {
        throw StringError("Bad argument");
      }
    }

    if (params) {
      if (!ERL_IS_LIST(params)) throw StringError("Bad argument");
      auto s = erl_length(params);
      for (int i = 0; i < s; i++) {
        auto param = erl_hd(params);
        params = erl_tl(params);
        if (ERL_IS_TUPLE(param)) {
          auto param_type  = erl_element(1, param);
          auto param_value = erl_element(2, param);
          if (ERL_IS_ATOM(param_type) && ATOM_EQ(param_type, "memory")) {
            auto mem = driver->GetMemory(Get<int>(param_value));
            if (!mem) throw StringError("Invalid memory handle");
            arguments->Add(*mem);
          }
        } else if (ERL_IS_INTEGER(param)) {
          arguments->Add(ERL_INT_VALUE(param));
        } else if (ERL_IS_FLOAT(param)) {
          float f = ERL_FLOAT_VALUE(param);
          arguments->Add(f);
        } else {
          throw StringError("Bad argument");
        }
      }
    }
  }
}

void RunCommand::Run(Context &ctx) {
  CUfunction func;
  CUresult result;

  result = cuModuleGetFunction(&func, ctx.module, kernel.c_str());
  if (result != CUDA_SUCCESS) throw DriverError(result);

  // DEBUG("Launch DriverPort::Stream");
  result = cuLaunchKernel(func, gx, gy, gz, bx, by, bz, 0, ctx.stream, arguments->GetPtr(), 0);
  // DEBUG("Exit DriverPort::Stream");
  if (result != CUDA_SUCCESS) throw DriverError(result, "Driver:execution");
  // DEBUG("Exit 1 DriverPort::Stream");
}

} // namespace Commands
