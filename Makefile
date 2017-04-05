MIX = mix
SOURCES = $(wildcard c_src/*.cpp)

GPU_DEBUG ?= "0"

# common c++ compiler flags
CXXFLAGS ?= -g -O3 -ansi -std=c++11 -pedantic -Wall -Wextra -Wno-long-long -DGPU_DEBUG=$(GPU_DEBUG)
LIBS =

# os specific flags
ifneq ($(OS), Windows_NT)
	CXXFLAGS += -fPIC

	ifeq ($(shell uname), Darwin)
		LDFLAGS += -dynamiclib -undefined dynamic_lookup
	endif
endif

# erl_interface library
EI_INCL  = $(shell erl -eval 'io:format("~s", [code:lib_dir(erl_interface, include)])' -s init stop -noshell)
EI_LIBS  = $(shell erl -eval 'io:format("~s", [code:lib_dir(erl_interface, lib)])' -s init stop -noshell)
CXXFLAGS += -I$(EI_INCL)
LIBS 		 += -L$(EI_LIBS) -lerl_interface -lei -lpthread

# CUDA library
CUDA     ?= "cuda"
CXXFLAGS += $(shell pkg-config --cflags $(CUDA))
LIBS     += $(shell pkg-config --libs-only-L $(CUDA)) -lcudart -lcuda

.PHONY: all port clean

all: port

port:
	$(MIX) compile

priv:
	mkdir -p priv

priv/cuda_port: priv $(SOURCES)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $(SOURCES) $(LIBS)

priv/cuda_port.exe: priv $(SOURCES)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $(SOURCES) $(LIBS)

clean:
	$(RM) -f priv/cuda_port
