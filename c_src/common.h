#ifndef __COMMON_H__
#define __COMMON_H__

#include <stdio.h>
#include <string>
#include <list>
#include <map>
#include <cstring>
#include <iostream>

extern "C" {
  #include "erl_interface.h"
  #include "ei.h"
}

#include "cuda.h"
#include "cuda_runtime.h"

#define C_STR(str) ((char*)std::string(str).c_str())
#define FORMAT(fmt, ...) erl_format(C_STR(fmt), ##__VA_ARGS__)
#define OK_STR C_STR("ok")
#define ERROR_STR C_STR("error")
#define MAKE_BINARY(str) erl_mk_binary(C_STR(str), sizeof(str) - 1)
#define ATOM_EQ(term, str) (strncmp(ERL_ATOM_PTR(term), str, sizeof(str) - 1) == 0)
#define IS_NIL(term) (ERL_IS_ATOM(term) && strncmp(ERL_ATOM_PTR(term), "nil", 3) == 0)
#define IS_OK_TUPLE(term) (strncmp(ERL_ATOM_PTR(erl_element(1, term)), "ok", 2) == 0)

class Error {
public:
  virtual ETERM *AsTerm() = 0;
};

class TermError : public Error {
public:
  ETERM *term;
  TermError(ETERM *error): Error(), term(error) {}
  virtual ETERM *AsTerm() { return term; }
};

class StringError : public Error {
public:
  std::string message;
  StringError(const char *errorMessage): Error(), message(errorMessage) {}
  virtual ETERM *AsTerm();
};

class RuntimeError : public Error {
public:
  cudaError_t code;
  RuntimeError(cudaError_t errorNo): Error(), code(errorNo) {}
  virtual ETERM *AsTerm();
};

class DriverError : public Error {
public:
  CUresult code;
  DriverError(CUresult errorNo): Error(), code(errorNo) {}
  virtual ETERM *AsTerm();
};

typedef std::map<std::string, ETERM *> Keywords;
Keywords GetKeywords(ETERM *list);

template <typename T> T Get(ETERM *);

#endif // __COMMON_H__
