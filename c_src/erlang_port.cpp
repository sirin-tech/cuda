#include "erlang_port.h"

using __gnu_cxx::stdio_filebuf;

// Swaps big-endian to little-endian or opposite
template <class T> void EndianSwap(T *buffer) {
  unsigned char *mem = reinterpret_cast<unsigned char *>(buffer);
  std::reverse(mem, mem + sizeof(T));
}

ErlangPort::ErlangPort(int device) :
    input(new stdio_filebuf<char>(PORTIN_FILENO, std::ios::in)),
    output(new stdio_filebuf<char>(PORTOUT_FILENO, std::ios::out)) {

  input.exceptions(std::ifstream::failbit | std::ifstream::badbit |
                   std::ifstream::eofbit);
  output.exceptions(std::ofstream::failbit | std::ofstream::badbit |
                    std::ofstream::eofbit);
  erl_init(NULL, 0);
  try {
    Init(this, device);
  } catch (Error &e) {
    result = e.AsTerm();
    WritePacket(result);
    throw StringError("Initializing error");
  }
  DEBUG("Port initialized");
}

ErlangPort::~ErlangPort() {
  DEBUG("Port destroyed");
  if (tuple) erl_free_compound(tuple);
  if (func) erl_free_term(func);
  if (arg) erl_free_term(arg);
  if (result) erl_free_term(result);
  if (driver) delete driver;
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

ETERM *ErlangPort::ReadPacket(uint32_t len) {
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

void ErlangPort::WritePacket(ETERM *packet) {
  auto len = erl_term_len(result);
  uint8_t type = TERM_PACKET;
  std::string buf(len, 0);
  erl_encode(packet, (unsigned char *)buf.c_str());
  WritePacketLength(len + 1);
  output.write((const char *)&type, 1);
  output.write(buf.c_str(), len);
  output.flush();
}

void ErlangPort::SendRawReply(void *data, size_t size) {
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
      tuple = ReadPacket(len - 1);
      if (!ERL_IS_TUPLE(tuple) || ERL_TUPLE_SIZE(tuple) != 2) continue;
      // Retrieve function name and argument
      func  = erl_element(1, tuple);
      arg   = erl_element(2, tuple);
      // If first element of tuple is not an atom - skip it
      if (!ERL_IS_ATOM(func)) continue;

      // First tuple element is atom
      std::string atomFunc(ERL_ATOM_PTR(func));
      // break on exit command
      if (atomFunc == "exit") break;
      // Search for registered functions
      auto handlerIt = handlers.find(atomFunc);
      // If there are no function to handle - skip packet
      if (handlerIt == handlers.end()) continue;
      // Handler founded - call it
      try {
        result = handlerIt->second(this, arg);
      } catch (Error &e) {
        result = e.AsTerm();
      }
    } else if (type == RAW_PACKET) {
      auto funcId = ReadRawFunc();
      std::shared_ptr<void> data(new char[len - 2]);
      input.read((char *)data.get(), len - 2);
      auto handlerIt = rawHandlers.find(funcId);
      if (handlerIt == rawHandlers.end()) continue;
      try {
        result = handlerIt->second(this, data, len - 2);
      } catch (Error &e) {
        result = e.AsTerm();
      }
    }

    if (result) {
      WritePacket(result);
    }
  };
}

void ErlangPort::AddHandler(std::string name, ErlangHandler handler) {
  handlers.insert(std::pair<std::string, ErlangHandler>(name, handler));
}

void ErlangPort::AddRawHandler(int id, ErlangRawHandler handler) {
  rawHandlers.insert(std::pair<int, ErlangRawHandler>(id, handler));
}

void ErlangPort::RemoveHandler(std::string name) {
  handlers.erase(name);
}

void ErlangPort::RemoveRawHandler(int id) {
  rawHandlers.erase(id);
}
