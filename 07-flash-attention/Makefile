NVCC = nvcc
NVCC_FLAGS = -O2 -lcublas

.PHONY: all clean

all: naive flash flash_causal

naive: exercises/01_attention_naive.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

flash: exercises/02_flash_forward.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

flash_causal: exercises/03_flash_causal.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

clean:
	rm -f naive flash flash_causal
