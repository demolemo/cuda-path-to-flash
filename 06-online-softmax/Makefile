NVCC = nvcc
NVCC_FLAGS = -O2

.PHONY: all clean

all: ex01 ex02 ex03

ex01: exercises/01_naive_softmax.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

ex02: exercises/02_online_softmax.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

ex03: exercises/03_fused_softmax.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

clean:
	rm -f ex01 ex02 ex03
