#ifndef __COMMON_H__
#define __COMMON_H__

#include <stdio.h>
#include <string>
#include <list>

extern "C" {
  #include "erl_interface.h"
  #include "ei.h"
}

#include "cuda.h"
#include "cuda_runtime.h"

#define MAKE_BINARY(str) erl_mk_binary(str, sizeof(str) - 1)
#define ATOM_EQ(term, str) (strncmp(ERL_ATOM_PTR(term), str, sizeof(str) - 1) == 0)
#define ERROR(str) erl_format("{~a~w}", "error", MAKE_BINARY(str))
#define IS_NIL(term) (ERL_IS_ATOM(term) && strncmp(ERL_ATOM_PTR(term), "nil", 3) == 0)
#define IS_OK_TUPLE(term) (strncmp(ERL_ATOM_PTR(erl_element(1, term)), "ok", 2) == 0)

class Error {
public:
  ETERM *message;
  Error(ETERM *errorMessage): message(errorMessage) {}
};

ETERM *CudaRuntimeError(cudaError_t error);
ETERM *CudaDriverError(CUresult error);

// API functions
ETERM *Info(ETERM *arg);

#endif // __COMMON_H__
