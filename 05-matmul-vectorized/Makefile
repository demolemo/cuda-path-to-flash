NVCC = nvcc
NVCC_FLAGS = -O2 -lcublas --ptxas-options=-v

.PHONY: all clean

all: k6 k7 k8

k6: exercises/01_vectorized.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

k7: exercises/02_resolve_conflicts.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

k8: exercises/03_warptiling.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

clean:
	rm -f k6 k7 k8
