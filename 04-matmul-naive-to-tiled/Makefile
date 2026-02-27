NVCC = nvcc
NVCC_FLAGS = -O2 -lcublas

.PHONY: all clean

all: k1 k2 k3 k4 k5

k1: exercises/01_naive.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

k2: exercises/02_coalesced.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

k3: exercises/03_shmem_tiled.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

k4: exercises/04_1d_blocktile.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

k5: exercises/05_2d_blocktile.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

clean:
	rm -f k1 k2 k3 k4 k5
