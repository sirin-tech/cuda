#include "common.h"
#include "erlang_driver.h"

int main(void) {
  ErlangDriver driver;
  // NOTE: add all functions here and in common.h
  driver.AddHandler("info", Info);
  try {
    // enter driver loop
    driver.Loop();
  } catch(...) {} // exit normally on any errors - just die
  return 0;
}
