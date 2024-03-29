# choose your compiler, e.g. gcc/clang
# example override to clang: make run CC=clang
CC = gcc-9

# the most basic way of building that is most likely to work on most systems
.PHONY: run
run: run.c
	$(CC) -O3 -o run run.c topo.c -lm
	$(CC) -O3 -o get_fct_mmf.out -fopenmp -mcmodel=large topo.c get_fct_mmf.c -lm
	$(CC) -O3 -fPIC -shared -o get_fct_mmf.so -fopenmp  -mcmodel=large topo.c get_fct_mmf.c
