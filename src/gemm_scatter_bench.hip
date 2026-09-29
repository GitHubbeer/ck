#include <ck/ck.hpp>
#include <ck/tensor_operation/gpu/device/device_gemm.hpp>
#include <ck/tensor_operation/gpu/device/tensor_layout.hpp>
#include <ck/tensor_operation/gpu/device/gemm_specialization.hpp>
#include <ck/tensor_operation/gpu/device/impl/device_gemm_xdl.hpp>
#include <ck/tensor_operation/gpu/grid/gridwise_gemm_xdlops_v2r3.hpp>
#include <ck/tensor_operation/gpu/element/element_wise_operation.hpp>

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using F64 = double;
using Col = ck::tensor_layout::gemm::ColumnMajor;
using Row = ck::tensor_layout::gemm::RowMajor;
using Pass = ck::tensor_operation::element_wise::PassThrough;
template <ck::index_t... I> using Seq = ck::Sequence<I...>;
constexpr auto GemmPad = ck::tensor_operation::device::GemmSpecialization::MNKPadding;

// The same gfx9 FP64 XDL tile used by CK's mk/kn/mn instance family.
using DeviceGemm = ck::tensor_operation::device::DeviceGemmXdl<
    F64, F64, F64, F64, Col, Row, Row, Pass, Pass, Pass, GemmPad,
    256, 128, 128, 4, 2, 16, 16, 4, 4,
    Seq<4, 64, 1>, Seq<0, 2, 1>, Seq<0, 2, 1>, 1, 2, 2, true,
    Seq<4, 64, 1>, Seq<0, 2, 1>, Seq<0, 2, 1>, 1, 2, 2, true, 7, 1>;
using Gridwise = typename DeviceGemm::template GridwiseGemmBase<
    decltype(DeviceGemm::WarpTileConfig64)>;
using Argument = typename Gridwise::Argument;
constexpr int TileM = 128;
constexpr int TileN = 128;

__device__ int find_block(const int* offsets, int count, int x) {
  int lo = 0, hi = count;
  while (lo < hi) {
    int mid = lo + (hi - lo) / 2;
    if (offsets[mid] <= x) lo = mid + 1;
    else hi = mid;
  }
  return lo - 1;
}

__global__ void scatter_reference(const double* dense_col_major, int m,
                                  double* const* blocks, const int* offsets,
                                  const int* lds, int block_count) {
  const int pos = blockIdx.x * blockDim.x + threadIdx.x;
  if (pos >= m * m) return;
  const int row = pos % m, col = pos / m;
  const int bi = find_block(offsets, block_count, row);
  const int bj = find_block(offsets, block_count, col);
  const int lr = row - offsets[bi], lc = col - offsets[bj];
  double* block = blocks[static_cast<long>(bi) * block_count + bj];
  if (bi >= bj) block[lr + static_cast<long>(lc) * lds[bj]] -= dense_col_major[pos];
  else block[static_cast<long>(lr) * lds[bi] + lc] -= dense_col_major[pos];
}

template <bool HasMainKLoop>
__global__ void ck_gemm_scatter_kernel(Argument arg, double* dense,
                                       double* const* blocks,
                                       const int* offsets, const int* lds,
                                       int block_count) {
  // Execute the public CK Gridwise GEMM tile, then consume its tile result
  // immediately in this CTA. dense is a workspace, never a separately
  // launched producer/consumer stage.
  const auto a_desc = Gridwise::MakeAGridDescriptor_K0_M_K1(
      arg.M, arg.MPadded, arg.K, arg.K0, arg.StrideA);
  const auto b_desc = Gridwise::MakeBGridDescriptor_K0_N_K1(
      arg.K, arg.N, arg.NPadded, arg.K0, arg.StrideB);
  const auto c_desc = Gridwise::MakeCGridDescriptor_M_N(
      arg.M, arg.MPadded, arg.N, arg.NPadded, arg.StrideC);
  __shared__ char smem[Gridwise::GetSharedMemoryNumberOfByte(ck::get_device_arch())];

  Gridwise::template Run<HasMainKLoop>(arg.p_a_grid, arg.p_b_grid, dense,
                                       smem, a_desc, b_desc, c_desc);
  __threadfence_block();
  __syncthreads();

  using TileMap = typename Gridwise::Block2CTileMap;
  const TileMap map{arg.M, arg.N};
  const auto tile = map.CalculateBottomIndex(
      ck::make_multi_index(static_cast<int>(blockIdx.x)));
  const int row0 = tile[ck::Number<0>{}] * TileM;
  const int col0 = tile[ck::Number<1>{}] * TileN;

  __shared__ int row_block[TileM], row_start[TileM], row_ld[TileM];
  __shared__ int col_block[TileN], col_start[TileN], col_ld[TileN];
  for (int x = threadIdx.x; x < TileM; x += blockDim.x) {
    const int r = row0 + x;
    const int bi = r < arg.M ? find_block(offsets, block_count, r) : -1;
    row_block[x] = bi;
    row_start[x] = bi >= 0 ? offsets[bi] : 0;
    row_ld[x] = bi >= 0 ? lds[bi] : 0;
  }
  for (int x = threadIdx.x; x < TileN; x += blockDim.x) {
    const int c = col0 + x;
    const int bj = c < arg.N ? find_block(offsets, block_count, c) : -1;
    col_block[x] = bj;
    col_start[x] = bj >= 0 ? offsets[bj] : 0;
    col_ld[x] = bj >= 0 ? lds[bj] : 0;
  }
  __syncthreads();

  for (int p = threadIdx.x; p < TileM * TileN; p += blockDim.x) {
    const int ri = p % TileM;
    const int cj = p / TileM;
    const int i = row0 + ri, j = col0 + cj;
    if (i >= arg.M || j >= arg.N) continue;
    const int bi = row_block[ri], bj = col_block[cj];
    if (bi < 0 || bj < 0) continue;
    double* block = blocks[static_cast<long>(bi) * block_count + bj];
    const int r = i - row_start[ri], c = j - col_start[cj];
    const double v = dense[static_cast<long>(i) * arg.StrideC + j];
    if (bi >= bj) block[r + static_cast<long>(c) * col_ld[cj]] -= v;
    else block[static_cast<long>(r) * row_ld[ri] + c] -= v;
  }
}

void check(hipError_t e, const char* what) {
  if (e != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}
void check_blas(rocblas_status s, const char* what) {
  if (s != rocblas_status_success) throw std::runtime_error(std::string(what) + " failed");
}

struct DeviceBuffer {
  void* p = nullptr;
  explicit DeviceBuffer(size_t bytes) { check(hipMalloc(&p, bytes), "hipMalloc"); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept : p(other.p) { other.p = nullptr; }
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) { if (p) (void)hipFree(p); p = other.p; other.p = nullptr; }
    return *this;
  }
  ~DeviceBuffer() { if (p) (void)hipFree(p); }
  template <typename T> T* as() { return static_cast<T*>(p); }
};
}  // namespace

int main(int argc, char** argv) try {
  int repeat = argc > 1 ? std::atoi(argv[1]) : 50;
  int device_id = argc > 2 ? std::atoi(argv[2]) : 0;
  if (repeat <= 0 || device_id < 0) throw std::runtime_error("usage: gemm_scatter_bench [repeat>0] [device>=0]");
  check(hipSetDevice(device_id), "hipSetDevice");
  hipDeviceProp_t prop{};
  check(hipGetDeviceProperties(&prop, device_id), "hipGetDeviceProperties");
  std::cout << "GPU: " << prop.name << " (" << prop.gcnArchName << ")\n";
  if (std::string(prop.gcnArchName).rfind("gfx942", 0) != 0)
    throw std::runtime_error("this benchmark is tuned for MI300A/gfx942");

  // Match cdls_backup/examples/rocm/test_scatter3_v2.cpp.
  const int H[] = {19,237,37,219,65,191,83,173,101,155,117,139,123,133,127,129};
  constexpr int blocks_n = sizeof(H) / sizeof(H[0]);
  std::vector<int> offsets(blocks_n), lds(blocks_n);
  int M = 0;
  for (int i = 0; i < blocks_n; ++i) { offsets[i] = M; lds[i] = M + H[i]; M += H[i]; }
  const int N = M, K = 511;
  std::cout << "M=N=" << M << " K=" << K << " block_count=" << blocks_n << '\n';

  std::mt19937_64 rng(123);
  std::uniform_real_distribution<double> dist(-0.5, 0.5);
  std::vector<double> A(static_cast<size_t>(M) * K), B(static_cast<size_t>(N) * K);
  for (auto& x : A) x = dist(rng);
  for (auto& x : B) x = dist(rng);

  // Host backing allocations reproduce the lower/upper block views.
  std::vector<std::vector<double>> L(blocks_n), U(blocks_n), L0, U0;
  for (int i = 0; i < blocks_n; ++i) {
    L[i].resize(static_cast<size_t>(M - offsets[i]) * H[i]);
    U[i].resize(static_cast<size_t>(M - offsets[i]) * H[i]);
    for (auto& x : L[i]) x = dist(rng);
    for (auto& x : U[i]) x = dist(rng);
  }
  L0 = L; U0 = U;
  std::vector<double*> block_ptrs(blocks_n * blocks_n);
  std::vector<DeviceBuffer> dL, dU;
  for (int i = 0; i < blocks_n; ++i) {
    dL.emplace_back(L[i].size() * sizeof(double));
    dU.emplace_back(U[i].size() * sizeof(double));
    check(hipMemcpy(dL[i].p, L[i].data(), L[i].size()*sizeof(double), hipMemcpyHostToDevice), "copy L");
    check(hipMemcpy(dU[i].p, U[i].data(), U[i].size()*sizeof(double), hipMemcpyHostToDevice), "copy U");
  }
  for (int i = 0; i < blocks_n; ++i) for (int j = 0; j < blocks_n; ++j) {
    if (i >= j) block_ptrs[i*blocks_n+j] = dL[j].as<double>() + offsets[i] - offsets[j];
    else block_ptrs[i*blocks_n+j] = dU[i].as<double>() + offsets[j] - offsets[i];
  }

  DeviceBuffer dA(A.size()*sizeof(double)), dB(B.size()*sizeof(double));
  DeviceBuffer dOffsets(offsets.size()*sizeof(int)), dLds(lds.size()*sizeof(int));
  DeviceBuffer dBlocks(block_ptrs.size()*sizeof(double*));
  DeviceBuffer dDense(static_cast<size_t>(M)*N*sizeof(double));
  DeviceBuffer dDenseRef(static_cast<size_t>(M)*N*sizeof(double));
  check(hipMemcpy(dA.p,A.data(),A.size()*sizeof(double),hipMemcpyHostToDevice),"copy A");
  check(hipMemcpy(dB.p,B.data(),B.size()*sizeof(double),hipMemcpyHostToDevice),"copy B");
  check(hipMemcpy(dOffsets.p,offsets.data(),offsets.size()*sizeof(int),hipMemcpyHostToDevice),"copy offsets");
  check(hipMemcpy(dLds.p,lds.data(),lds.size()*sizeof(int),hipMemcpyHostToDevice),"copy lds");
  check(hipMemcpy(dBlocks.p,block_ptrs.data(),block_ptrs.size()*sizeof(double*),hipMemcpyHostToDevice),"copy pointers");

  // A is column-major MxK. B's bytes are row-major KxN, equivalent to the
  // column-major NxK input expected by the reference's transposed GEMM.
  Argument arg{dA.as<double>(), dB.as<double>(), dDense.as<double>(),
               M,N,K,M,N,N};
  if (!Gridwise::CheckValidity(arg))
    throw std::runtime_error("CK Gridwise GEMM does not support this shape/configuration");
  const int grid = ((M + TileM - 1)/TileM) * ((N + TileN - 1)/TileN);
  const auto launch_fused = [&] {
    if (Gridwise::CalculateHasMainKBlockLoop(K))
      hipLaunchKernelGGL((ck_gemm_scatter_kernel<true>), dim3(grid), dim3(256), 0, 0,
                         arg, dDense.as<double>(), dBlocks.as<double*>(),
                         dOffsets.as<int>(), dLds.as<int>(), blocks_n);
    else
      hipLaunchKernelGGL((ck_gemm_scatter_kernel<false>), dim3(grid), dim3(256), 0, 0,
                         arg, dDense.as<double>(), dBlocks.as<double*>(),
                         dOffsets.as<int>(), dLds.as<int>(), blocks_n);
    check(hipGetLastError(), "launch fused CK GEMM-scatter");
  };

  rocblas_handle blas;
  check_blas(rocblas_create_handle(&blas), "rocblas_create_handle");
  check_blas(rocblas_set_stream(blas, nullptr), "rocblas_set_stream");
  auto launch_reference = [&] {
    const double alpha = 1.0, beta = 0.0;
    check_blas(rocblas_dgemm(blas, rocblas_operation_none, rocblas_operation_transpose,
                             M, N, K, &alpha, dA.as<double>(), M,
                             dB.as<double>(), N, &beta, dDenseRef.as<double>(), M),
               "rocblas_dgemm");
    hipLaunchKernelGGL(scatter_reference, dim3((M*N+255)/256), dim3(256), 0, 0,
                       dDenseRef.as<double>(), M, dBlocks.as<double*>(),
                       dOffsets.as<int>(), dLds.as<int>(), blocks_n);
    check(hipGetLastError(), "launch reference scatter");
  };
  auto reset = [&] {
    for (int i=0;i<blocks_n;++i) {
      check(hipMemcpyAsync(dL[i].p,L0[i].data(),L0[i].size()*sizeof(double),hipMemcpyHostToDevice), "reset L");
      check(hipMemcpyAsync(dU[i].p,U0[i].data(),U0[i].size()*sizeof(double),hipMemcpyHostToDevice), "reset U");
    }
  };
  auto time_call = [&](auto&& fn) {
    hipEvent_t start, stop;
    check(hipEventCreate(&start), "create start event");
    check(hipEventCreate(&stop), "create stop event");
    fn(); check(hipDeviceSynchronize(), "warmup sync");
    float total = 0;
    for (int i=0;i<repeat;++i) {
      reset();
      check(hipEventRecord(start), "record start");
      fn();
      check(hipEventRecord(stop), "record stop");
      check(hipEventSynchronize(stop), "sync stop");
      float ms = 0; check(hipEventElapsedTime(&ms,start,stop), "elapsed time");
      total += ms;
    }
    (void)hipEventDestroy(start); (void)hipEventDestroy(stop);
    return total/repeat;
  };

  reset(); launch_fused(); check(hipDeviceSynchronize(), "fused correctness sync");
  std::vector<std::vector<double>> fusedL(blocks_n), fusedU(blocks_n);
  for (int i=0;i<blocks_n;++i) {
    fusedL[i].resize(L[i].size()); fusedU[i].resize(U[i].size());
    check(hipMemcpy(fusedL[i].data(),dL[i].p,L[i].size()*sizeof(double),hipMemcpyDeviceToHost),"read fused L");
    check(hipMemcpy(fusedU[i].data(),dU[i].p,U[i].size()*sizeof(double),hipMemcpyDeviceToHost),"read fused U");
  }
  reset(); launch_reference(); check(hipDeviceSynchronize(), "reference correctness sync");
  double max_diff = 0;
  for (int i=0;i<blocks_n;++i) {
    check(hipMemcpy(L[i].data(),dL[i].p,L[i].size()*sizeof(double),hipMemcpyDeviceToHost),"read ref L");
    check(hipMemcpy(U[i].data(),dU[i].p,U[i].size()*sizeof(double),hipMemcpyDeviceToHost),"read ref U");
    for (size_t x=0;x<L[i].size();++x) max_diff=std::max(max_diff,std::abs(L[i][x]-fusedL[i][x]));
    for (size_t x=0;x<U[i].size();++x) max_diff=std::max(max_diff,std::abs(U[i][x]-fusedU[i][x]));
  }
  std::cout << "max_abs_diff=" << max_diff << '\n';
  if (max_diff > 1e-10) throw std::runtime_error("fused result differs from rocBLAS+scatter");

  const float fused_ms = time_call(launch_fused);
  const float reference_ms = time_call(launch_reference);
  std::cout << "fused_ms,rocblas_plus_scatter_ms,speedup\n"
            << fused_ms << ',' << reference_ms << ',' << reference_ms/fused_ms << '\n';
  check_blas(rocblas_destroy_handle(blas), "rocblas_destroy_handle");
  return 0;
} catch (const std::exception& e) {
  std::cerr << "error: " << e.what() << '\n';
  return 1;
}
