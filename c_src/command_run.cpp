#include "common.h"
#include "driver.h"
#include "erlang_port.h"

ETERM *Run(ErlangPort *port, ETERM *arg) {
  if (!port->driver) throw StringError("Driver not initialized");
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
  RunParameters args;

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
    if (params) {
      if (!ERL_IS_LIST(params)) throw StringError("Bad argument");
      auto s = erl_length(params);
      for (int i = 0; i < s; i++) {
        auto param = erl_hd(params);
        params = erl_tl(params);
        if (ERL_IS_TUPLE(param)) {
          auto param_type = erl_element(1, param);
          auto param_value = erl_element(2, param);
          if (ERL_IS_ATOM(param_type) && ATOM_EQ(param_type, "memory")) {
            auto mem = port->driver->GetMemory(Get<int>(param_value));
            if (!mem) throw StringError("Invalid memory handle");
            args.Add(*mem);
          }
        } else if (ERL_IS_INTEGER(param)) {
          args.Add(ERL_INT_VALUE(param));
        } else if (ERL_IS_FLOAT(param)) {
          float f = ERL_FLOAT_VALUE(param);
          args.Add(f);
        } else {
          throw StringError("Bad argument");
        }
      }
    }
  }

  port->driver->Run(module, func, gx, gy, gz, bx, by, bz, args);
  return erl_mk_atom(OK_STR);
}
