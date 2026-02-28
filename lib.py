"""
Shared utilities for cuda-path-to-flash exercises.

Handles CUDA kernel compilation via PyTorch's load_inline,
benchmarking, and validation helpers.
"""

import os
import time
import torch
from torch.utils.cpp_extension import load_inline

# Module name counter to avoid recompilation conflicts
_module_counter = 0


def compile_cuda(cuda_src: str, functions: list[str], name: str = None,
                 extra_cuda_cflags: list[str] = None) -> object:
    """
    Compile CUDA source and return a module with callable functions.

    Args:
        cuda_src: CUDA C++ source code string
        functions: list of function names to expose (must have torch::Tensor args)
        name: module name (auto-generated if None)
        extra_cuda_cflags: additional nvcc flags

    Returns:
        Python module with the compiled functions
    """
    global _module_counter
    if name is None:
        _module_counter += 1
        name = f"cuda_module_{_module_counter}"

    cpp_src = ""
    for fn in functions:
        cpp_src += f"torch::Tensor {fn}(torch::Tensor input);\n"

    # We'll use the cuda_sources approach — function signatures
    # are picked up from the CUDA source automatically
    module = load_inline(
        name=name,
        cpp_sources="",  # empty — all in CUDA
        cuda_sources=cuda_src,
        functions=functions,
        extra_cuda_cflags=extra_cuda_cflags or ["-O2"],
        verbose=False,
    )
    return module


def compile_cuda_raw(cuda_src: str, name: str = None,
                     extra_cuda_cflags: list[str] = None) -> object:
    """
    Compile CUDA source with TORCH_LIBRARY registration.
    Use torch.ops.<module_name>.<function> to call.

    This is the pattern Gau Nernst uses — more flexible,
    supports arbitrary function signatures.
    """
    global _module_counter
    if name is None:
        _module_counter += 1
        name = f"cuda_raw_{_module_counter}"

    load_inline(
        name=name,
        cpp_sources="",
        cuda_sources=cuda_src,
        verbose=False,
        is_python_module=False,
        extra_cuda_cflags=extra_cuda_cflags or ["-O2"],
    )


def check_close(actual: torch.Tensor, expected: torch.Tensor,
                name: str = "", atol: float = 1e-4, rtol: float = 1e-4) -> bool:
    """Check if two tensors are close, print result."""
    try:
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        print(f"  ✅ {name}" if name else "  ✅ PASSED")
        return True
    except AssertionError as e:
        max_diff = (actual - expected).abs().max().item()
        print(f"  ❌ {name} (max_diff={max_diff:.6e})" if name else f"  ❌ FAILED (max_diff={max_diff:.6e})")
        # Print first few mismatches
        mask = (actual - expected).abs() > atol
        if mask.any():
            idxs = mask.nonzero()[:5]
            for idx in idxs:
                idx = tuple(idx.tolist())
                print(f"     [{idx}] got={actual[idx].item():.6f} expected={expected[idx].item():.6f}")
        return False


def bench(fn, *args, warmup: int = 10, iters: int = 100, label: str = ""):
    """Benchmark a CUDA function using torch.cuda.Event for accurate timing."""
    if os.environ.get("CUDA_BENCH") != "1":
        return None

    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn(*args)
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end) / iters
    if label:
        print(f"  ⏱  {label}: {ms:.3f} ms")
    return ms


def bench_compare(fns: dict[str, tuple], warmup=10, iters=100):
    """Benchmark multiple functions and show comparison.

    Args:
        fns: dict of {name: (fn, *args)}
    """
    if os.environ.get("CUDA_BENCH") != "1":
        return

    print("\n  Benchmarks:")
    results = {}
    for name, (fn, *args) in fns.items():
        ms = bench(fn, *args, warmup=warmup, iters=iters, label=name)
        if ms is not None:
            results[name] = ms

    if len(results) > 1:
        fastest = min(results.values())
        for name, ms in results.items():
            ratio = ms / fastest
            print(f"    {name}: {ratio:.2f}x {'(fastest)' if ratio == 1.0 else ''}")


def gpu_info():
    """Print basic GPU info."""
    if not torch.cuda.is_available():
        print("❌ No CUDA GPU available!")
        return False
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}")
    print(f"  SMs: {props.multi_processor_count}, "
          f"Compute: {props.major}.{props.minor}, "
          f"Memory: {props.total_mem / 1e9:.1f} GB")
    return True
