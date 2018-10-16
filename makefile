CC=gcc
STD=-std=c11
LFLAGS=
OFLAGS=-O2

.PHONY: all

all: heat_diffusion debug_heat

heat_diffusion: heat_diffusion.c
	$(CC) $(STD) $(OFLAGS) -o $@ $<

debug_heat: heat_diffusion.c
	$(CC) $(STD) -g -o $@ $<
