#ifndef __COMMON_H__
#define __COMMON_H__

#include <stdio.h>
#include <string>
#include <list>
#include <map>
#include <cstring>

#if GPU_DEBUG > 0
  #include <iostream>
#endif

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
#if GPU_DEBUG > 0
  #define DEBUG(msg) std::cout << msg << "\n"
#else
  #define DEBUG(msg) do {} while(0)
#endif

class Error {
protected:
  std::string source;
public:
  Error(const char *src = NULL) : source(src ? src : "") {}
  virtual ETERM *AsTerm() = 0;
};

class TermError : public Error {
public:
  ETERM *term;
  TermError(ETERM *error, const char *src = NULL): Error(src), term(error) {}
  virtual ETERM *AsTerm() { return term; }
};

class StringError : public Error {
public:
  std::string message;
  StringError(const char *errorMessage, const char *src = NULL): Error(src), message(errorMessage) {}
  virtual ETERM *AsTerm();
};

class RuntimeError : public Error {
public:
  cudaError_t code;
  RuntimeError(cudaError_t errorNo, const char *src = NULL): Error(src), code(errorNo) {}
  virtual ETERM *AsTerm();
};

class DriverError : public Error {
public:
  CUresult code;
  DriverError(CUresult errorNo, const char *src = NULL): Error(src), code(errorNo) {}
  virtual ETERM *AsTerm();
};

typedef std::map<std::string, ETERM *> Keywords;
Keywords GetKeywords(ETERM *list);

template <typename T> T Get(ETERM *);
int GetModuleIndex(ETERM *);
int GetMemoryIndex(ETERM *);

#endif // __COMMON_H__
