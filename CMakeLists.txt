cmake_minimum_required(VERSION 3.24)
project(ck_gemm_scatter LANGUAGES CXX HIP)

set(CK_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/composable_kernel" CACHE PATH "Composable Kernel source tree")
if(NOT EXISTS "${CK_ROOT}/include/ck/ck.hpp")
  message(FATAL_ERROR "Set CK_ROOT to a Composable Kernel checkout")
endif()

# CK's generated config is needed even for header-only GridwiseGemm use.
set(DTYPES "fp64")
set(CK_ENABLE_FP64 ON)
set(CK_USE_XDL ON)
set(CK_USE_GFX94 ON)
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated/ck")
configure_file("${CK_ROOT}/include/ck/config.h.in"
               "${CMAKE_CURRENT_BINARY_DIR}/generated/ck/config.h")

find_package(hip REQUIRED CONFIG)
find_package(rocblas REQUIRED CONFIG)

add_executable(gemm_scatter_bench src/gemm_scatter_bench.hip)
target_compile_features(gemm_scatter_bench PRIVATE cxx_std_17)
target_include_directories(gemm_scatter_bench PRIVATE
  "${CMAKE_CURRENT_BINARY_DIR}/generated"
  "${CK_ROOT}/include")
target_compile_options(gemm_scatter_bench PRIVATE
  $<$<COMPILE_LANGUAGE:HIP>:-O3>)
set_property(TARGET gemm_scatter_bench PROPERTY HIP_ARCHITECTURES gfx942)
target_link_libraries(gemm_scatter_bench PRIVATE roc::rocblas hip::host)
