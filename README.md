# CK FP64 GEMM + scatter for MI300A

This prototype targets MI300A (`gfx942`) and uses CK's `DeviceGemmXdl` gridwise
tile for FP64 GEMM. Each CTA writes its computed tile into a reusable dense
workspace and then scatters that tile into the block storage before exiting.
The GEMM and scatter work therefore share one kernel launch; the dense
workspace remains an intermediate global-memory round trip and is included in
the measured fused time.

The benchmark reproduces the block partition and `K=511` case from
`cdls_backup/examples/rocm/test_scatter3_v2.cpp`. It checks output against
rocBLAS dgemm plus the reference scatter and reports both elapsed times and
their ratio.

## Build

Requirements: ROCm HIP compiler/runtime and rocBLAS. The CK headers needed by
this prototype are included under `third_party/composable_kernel/include` from
upstream revision `2b053c6c5e60efc86cc62b12a9ca490ee2d5ebcf`; CMake generates
CK's config header. Set `CK_ROOT` only if you want to use a separate CK source
checkout.

```sh
conda create -y -p ./.conda python=3.12 cmake ninja  # optional, if CMake is absent
conda activate ./.conda
cmake -S . -B build-gfx942 \
  -DCMAKE_HIP_COMPILER=/path/to/rocm/llvm/bin/clang++ \
  -DCMAKE_HIP_COMPILER_ROCM_ROOT=/path/to/rocm \
  -DCMAKE_HIP_ARCHITECTURES=gfx942
cmake --build build-gfx942 -j
```

For this container the HIP compiler is `/usr/lib64/rocm/llvm/bin/clang++` and
the ROCm root is `/usr`.

## Run

```sh
./build-gfx942/gemm_scatter_bench [repeat=50] [device=0]
```

The executable reports `max_abs_diff` and a CSV row containing fused time,
rocBLAS-plus-scatter time, and speedup. Run it on an MI300A host; the local
development GPU is a gfx1201 card and cannot provide MI300A performance data.
