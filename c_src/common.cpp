#include "common.h"

ETERM *StringError::AsTerm() {
  return FORMAT("{~a,~w}", ERROR_STR, erl_mk_binary(message.c_str(), message.size()));
}

ETERM *RuntimeError::AsTerm() {
  const char *error = cudaGetErrorString(code);
  return FORMAT("{~a,~w}", ERROR_STR, erl_mk_binary(error, strlen(error)));
}

ETERM *DriverError::AsTerm() {
  const char *buf;
  if (cuGetErrorString(code, &buf) == CUDA_SUCCESS) {
    return FORMAT("{~a,~w}", ERROR_STR, erl_mk_binary(buf, strlen(buf)));
  }
  return FORMAT("{~a,~w}", ERROR_STR, MAKE_BINARY("Unknown error"));
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
