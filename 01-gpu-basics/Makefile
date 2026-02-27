NVCC = nvcc
NVCC_FLAGS = -O2

.PHONY: all test bench clean

all: test bench

# --- Exercises (compile individually) ---
ex01: exercises/01_hello_gpu.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

ex02: exercises/02_vector_add.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

ex03: exercises/03_matrix_ops.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

exercises: ex01 ex02 ex03

# --- Tests ---
test: tests/test_01.cu
	$(NVCC) $(NVCC_FLAGS) -o test_01 $<
	./test_01

# --- Benchmarks ---
bench: benchmarks/bench.cu
	$(NVCC) $(NVCC_FLAGS) -o bench_01 $<
	./bench_01

clean:
	rm -f ex01 ex02 ex03 test_01 bench_01
