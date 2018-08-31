#!/bin/make
OPT := -Ofast -march=native -mfpmath=both

GTK := `pkg-config --cflags --libs gtk+-3.0`
PULSE := `pkg-config --cflags --libs libpulse`
CXXFLAGS := $(OPT) -std=c++1z -Wall -I/usr/local/cuda/include $(GTK) $(PULSE)

LDFLAGS := -L/usr/local/cuda/lib64 $(GTK) $(PULSE)
LIBS := -lcudart

NVCC := nvcc
NVCCFLAGS := -O3 -I ~/NVIDIA_CUDA-9.2_Samples/common/inc/ -gencode=arch=compute_60,code=sm_60


PROGRAM := sound_split
CXXSRC := main.cpp converter.cpp spec_convert.cpp player.cpp filterspec_convert.cpp
CUSRC := analyse.cu wave.cu findphase.cu


CXXDEPS := $(CXXSRC:%.cpp=%.d)
CXXOBJS := $(CXXSRC:%.cpp=%.o)
CUOBJS := $(CUSRC:%.cu=%.o)
OBJS := $(CXXOBJS) $(CUOBJS)

all: ${PROGRAM}

$(PROGRAM): $(OBJS)
	$(CXX) $(OPT) -o $@ $^ $(LDFLAGS) $(LIBS)

.SUFFIXES: .cu

$(CUOBJS): %.o: %.cu
	$(NVCC) -c $(NVCCFLAGS) $< -o $@

${CXXDEPS}: %.d: %.cpp
	$(CXX) $< -MM -MP -MF $@

$(CXXOBJS): %.o: %.cpp %.d
	$(CXX) $(CXXFLAGS) -c $< -o $@

wave.o: wave.hpp fmrs.hpp

analyse.o: analyse.hpp fmrs.hpp

findphase.o: findphase.hpp fmrs.hpp

-include $(CXXDEPS)

clean:
	$(RM) $(PROGRAM)
	$(RM) $(OBJS)
	$(RM) $(CXXDEPS)

.PHONY: all clean

#OLD_SHELL := $(SHELL)
#SHELL = $(warning [Making: $@] [Dependencies: $^] [Changed: $?])$(OLD_SHELL)
