CC=gcc
STD=-std=c18
LIBS=-lm -lOpenCL
OFLAGS=-O2
LDIRS=-L/usr/lib64/
IDIRS=-I/usr/include/
ARCH=-march=native

.PHONY: all

all: heat_diffusion debug_heat

heat_diffusion: heat_diffusion.c
	$(CC) $(LDIRS) $(IDIRS) $(LIBS) $(STD) $(OFLAGS) -o $@ $< $(ARCH)

debug_heat: heat_diffusion.c
	$(CC) $(LDIRS) $(IDIRS) $(LIBS) $(STD) -g -o $@ $< -L$(LDIRS) $(ARCH)
