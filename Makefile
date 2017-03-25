MIX = mix
SOURCES = $(wildcard c_src/*.cpp)

# common c++ compiler flags
CXXFLAGS ?= -g -O3 -ansi -std=c++11 -pedantic -Wall -Wextra -Wno-long-long -Wno-write-strings
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
LIBS     += $(shell pkg-config --libs-only-L $(CUDA)) -lcudart

.PHONY: all port clean

all: port

port:
	$(MIX) compile

priv:
	mkdir -p priv

priv/gpu_math_port: priv clean $(SOURCES)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $(SOURCES) $(LIBS)

priv/gpu_math_port.exe: priv clean $(SOURCES)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $(SOURCES) $(LIBS)

clean:
	$(RM) -r priv/gpu_math_port
