NVCC = nvcc
NVCC_FLAGS = -O2

.PHONY: all exercises clean

all: exercises

ex01: exercises/01_multi_block_reduce.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

ex02: exercises/02_scan.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

ex03: exercises/03_histogram.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

exercises: ex01 ex02 ex03

clean:
	rm -f ex01 ex02 ex03
