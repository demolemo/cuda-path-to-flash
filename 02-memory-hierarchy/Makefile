NVCC = nvcc
NVCC_FLAGS = -O2

.PHONY: all exercises test clean

all: exercises

ex01: exercises/01_shmem_reduce.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

ex02: exercises/02_transpose.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

ex03: exercises/03_tiled_matmul.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

exercises: ex01 ex02 ex03

clean:
	rm -f ex01 ex02 ex03
