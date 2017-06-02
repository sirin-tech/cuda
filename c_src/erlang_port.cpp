#include "erlang_port.h"

using __gnu_cxx::stdio_filebuf;

// Swaps big-endian to little-endian or opposite
template <class T> void EndianSwap(T *buffer) {
  unsigned char *mem = reinterpret_cast<unsigned char *>(buffer);
  std::reverse(mem, mem + sizeof(T));
}

ErlangPort::ErlangPort() :
    input(new stdio_filebuf<char>(PORTIN_FILENO, std::ios::in)),
    output(new stdio_filebuf<char>(PORTOUT_FILENO, std::ios::out)) {
  input.exceptions(std::ifstream::failbit | std::ifstream::badbit |
                   std::ifstream::eofbit);
  output.exceptions(std::ofstream::failbit | std::ofstream::badbit |
                    std::ofstream::eofbit);
  erl_init(NULL, 0);
  DEBUG("Port initialized");
}

ErlangPort::~ErlangPort() {
  DEBUG("Port destroyed");
  if (tuple) erl_free_compound(tuple);
  if (funcAtom) erl_free_term(funcAtom);
  if (arg) erl_free_term(arg);
  if (result) erl_free_term(result);
}

uint32_t ErlangPort::ReadPacketLength() {
  uint32_t len;
  input.read(reinterpret_cast<char*>(&len), sizeof(len));
  EndianSwap(&len);
  return len;
}

uint8_t ErlangPort::ReadPacketType() {
  uint8_t type;
  input.read(reinterpret_cast<char*>(&type), sizeof(type));
  return type;
}

uint8_t ErlangPort::ReadRawFunc() {
  uint8_t func;
  input.read(reinterpret_cast<char*>(&func), sizeof(func));
  return func;
}

ETERM *ErlangPort::ReadTermPacket(uint32_t len) {
  // Read packet data, len bytes
  std::string buf(len, 0);
  input.read((char *)buf.c_str(), len);
  // Decode packet
  return erl_decode((unsigned char *)buf.c_str());
}

void ErlangPort::WritePacketLength(uint32_t len) {
  EndianSwap(&len);
  output.write(reinterpret_cast<char*>(&len), sizeof(len));
}

#define TERM_PACKET 1
#define RAW_PACKET 2

void ErlangPort::WriteTermPacket(ETERM *packet) {
  auto len = erl_term_len(packet);
  uint8_t type = TERM_PACKET;
  std::string buf(len, 0);
  erl_encode(packet, (unsigned char *)buf.c_str());
  WritePacketLength(len + 1);
  output.write((const char *)&type, 1);
  output.write(buf.c_str(), len);
  output.flush();
}

void ErlangPort::WriteRawPacket(void *data, size_t size) {
  uint8_t type = RAW_PACKET;
  WritePacketLength(size + 1);
  output.write((const char *)&type, 1);
  output.write((const char *)data, size);
  output.flush();
}

void ErlangPort::Loop() {
  while(true) {
    // Read packet length, 4 bytes
    auto len = ReadPacketLength();
    auto type = ReadPacketType();
    // ErlangHandler handler = NULL;
    result = NULL;
    if (type == TERM_PACKET) {
      tuple = ReadTermPacket(len - 1);
      if (!ERL_IS_TUPLE(tuple) || ERL_TUPLE_SIZE(tuple) != 2) continue;
      // Retrieve function name and argument
      funcAtom = erl_element(1, tuple);
      arg      = erl_element(2, tuple);
      // If first element of tuple is not an atom - skip it
      if (!ERL_IS_ATOM(funcAtom)) continue;
      std::string termFunc(ERL_ATOM_PTR(funcAtom));
      if (termFunc == "exit") break;
      // handle request
      try {
        result = HandleTermFunction(termFunc, arg);
      } catch (Error &error) {
        result = error.AsTerm();
      }
    } else if (type == RAW_PACKET) {
      // read size of function name
      uint8_t funcSize = 0;
      input.read((char *)&funcSize, 1);
      if (funcSize == 0) continue;
      // read function name
      std::string rawFunc(funcSize, 0);
      input.read((char *)rawFunc.c_str(), funcSize);
      if (rawFunc == "exit") break;
      // read raw data
      len = len - 2 - funcSize;
      std::shared_ptr<void> data(new char[len]);
      input.read((char *)data.get(), len);
      // handle request
      try {
        result = HandleRawFunction(rawFunc, data, len);
      } catch (Error &error) {
        result = error.AsTerm();
      }
    }

    if (result) {
      WriteTermPacket(result);
    }
  };
}
