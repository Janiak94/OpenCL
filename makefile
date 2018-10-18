CC=gcc
STD=-std=c99
LIBS=-lm -lOpenCL -lrt
OFLAGS=-O2
LDIRS=-L/usr/lib64/
IDIRS=-I/usr/include/

.PHONY: all

all: heat_diffusion debug_heat

heat_diffusion: heat_diffusion.c
	$(CC) $(LDIRS) $(IDIRS) $(LIBS) $(STD) $(OFLAGS) -o $@ $<

debug_heat: heat_diffusion.c
	$(CC) $(LDIRS) $(IDIRS) $(LIBS) $(STD) -g -o $@ $< -L$(LDIRS)
