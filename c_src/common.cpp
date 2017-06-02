#include "common.h"

ETERM *StringError::AsTerm() {
  return FORMAT("{error,~w,~w}",
    erl_mk_binary(source.c_str(), source.size()),
    erl_mk_binary(message.c_str(), message.size()));
}

ETERM *RuntimeError::AsTerm() {
  const char *name = cudaGetErrorName(code);
  const char *str = cudaGetErrorString(code);
  return FORMAT("{error,~w,~w,~w}",
    erl_mk_binary(source.c_str(), source.size()),
    erl_mk_binary(name, strlen(name)),
    erl_mk_binary(str, strlen(str)));
}

ETERM *DriverError::AsTerm() {
  const char *name, *str;
  if (cuGetErrorName(code, &name) == CUDA_SUCCESS &&
      cuGetErrorString(code, &str) == CUDA_SUCCESS) {
    // DEBUG("DeviceError: " << name << ", " << str);
    return FORMAT("{error,~w,~w,~w}",
      erl_mk_binary(source.c_str(), source.size()),
      erl_mk_binary(name, strlen(name)),
      erl_mk_binary(str, strlen(str)));
  }
  return FORMAT("{error,~w,~w}",
    erl_mk_binary(source.c_str(), source.size()),
    MAKE_BINARY("Unknown error"));
}

Keywords GetKeywords(ETERM *list) {
  if (!ERL_IS_LIST(list)) throw StringError("Bad argument");
  Keywords map;
  auto size = erl_length(list);
  for (int i = 0; i < size; i++) {
    auto tuple = erl_hd(list);
    list = erl_tl(list);
    if (!ERL_IS_TUPLE(tuple) || erl_size(tuple) != 2) throw StringError("Bad argument");
    auto keyAtom = erl_element(1, tuple);
    if (!ERL_IS_ATOM(keyAtom)) throw StringError("Bad argument");
    std::string key(ERL_ATOM_PTR(keyAtom));
    map.insert(std::pair<std::string, ETERM *>(key, erl_element(2, tuple)));
  }
  return map;
}

template <> int Get<int>(ETERM *value) {
  if (!ERL_IS_INTEGER(value)) throw StringError("Bad argument");
  return ERL_INT_VALUE(value);
}

template <> unsigned int Get<unsigned int>(ETERM *value) {
  if (!ERL_IS_INTEGER(value)) throw StringError("Bad argument");
  return ERL_INT_UVALUE(value);
}

template <> bool Get<bool>(ETERM *value) {
  if (!ERL_IS_ATOM(value)) throw StringError("Bad argument");
  if (ATOM_EQ(value, "true")) return true;
  if (ATOM_EQ(value, "false")) return false;
  throw StringError("Bad argument");
}

int GetModuleIndex(ETERM *value) {
  if (!ERL_IS_TUPLE(value) || erl_size(value) != 2) {
    throw StringError("Invalid module handle");
  }
  auto a = erl_element(1, value);
  auto v = erl_element(2, value);
  if (!ERL_IS_ATOM(a) || !ATOM_EQ(a, "module")) {
    throw StringError("Invalid module handle");
  }
  return Get<int>(v);
}

int GetMemoryIndex(ETERM *value) {
  if (!ERL_IS_TUPLE(value) || erl_size(value) != 2) {
    throw StringError("Invalid memory handle");
  }
  auto a = erl_element(1, value);
  auto v = erl_element(2, value);
  if (!ERL_IS_ATOM(a) || !ATOM_EQ(a, "memory")) {
    throw StringError("Invalid memory handle");
  }
  return Get<int>(v);
}
