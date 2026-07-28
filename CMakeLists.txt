cmake_minimum_required(VERSION 3.10)

project(RaBitQLib LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenMP REQUIRED)

option(RABITQ_BUILD_PYTHON_BINDINGS "Build Python bindings" OFF)
option(RABITQ_BUILD_TESTS "Build tests" OFF)
option(RABITQ_BUILD_BENCHMARKS "Build deterministic SIMD benchmarks" OFF)

set(RABITQ_TARGET_PROCESSOR "${CMAKE_SYSTEM_PROCESSOR}")
if(APPLE AND CMAKE_OSX_ARCHITECTURES)
    list(LENGTH CMAKE_OSX_ARCHITECTURES RABITQ_OSX_ARCH_COUNT)
    if(NOT RABITQ_OSX_ARCH_COUNT EQUAL 1)
        message(FATAL_ERROR "RaBitQ builds one native architecture per artifact; Universal Binary is unsupported")
    endif()
    set(RABITQ_TARGET_PROCESSOR "${CMAKE_OSX_ARCHITECTURES}")
endif()
string(TOLOWER "${RABITQ_TARGET_PROCESSOR}" RABITQ_TARGET_PROCESSOR)

set(RABITQ_COMMON_SOURCES
    src/index/hnsw_search_dispatch.cpp
    src/utils/cpu_features.cpp
    src/simd/dispatch.cpp
)

set(RABITQ_AVX2_SOURCES
    src/simd/pack_excode_avx2.cpp
    src/simd/space_excode_avx2.cpp
    src/simd/space_avx2.cpp
    src/simd/fastscan_avx2.cpp
    src/simd/warmup_avx2.cpp
    src/simd/rotator_avx2.cpp
)

set(RABITQ_AVX512_SOURCES
    src/simd/pack_excode_avx512.cpp
    src/simd/space_excode_avx512.cpp
    src/simd/space_avx512.cpp
    src/simd/fastscan_avx512.cpp
    src/simd/rotator_avx512.cpp
)

set(RABITQ_AVX512_POPCNT_SOURCES
    src/simd/warmup_avx512.cpp
)

set(RABITQ_HNSW_AVX2_SOURCES
    src/index/hnsw_search_avx2.cpp
)

set(RABITQ_HNSW_AVX512_CORE_SOURCES
    src/index/hnsw_search_avx512_core.cpp
)

set(RABITQ_HNSW_AVX512_POPCNT_SOURCES
    src/index/hnsw_search_avx512_popcnt.cpp
)

set(RABITQ_NEON_SOURCES
    src/simd/pack_excode_neon.cpp
    src/simd/space_excode_neon.cpp
    src/simd/space_neon.cpp
    src/simd/fastscan_neon.cpp
    src/simd/warmup_neon.cpp
    src/simd/rotator_neon.cpp
    src/index/hnsw_search_neon.cpp
)

set(RABITQ_ARCH_SOURCES)
if(RABITQ_TARGET_PROCESSOR MATCHES "^(x86_64|amd64)$")
    list(APPEND RABITQ_ARCH_SOURCES
        ${RABITQ_AVX2_SOURCES}
        ${RABITQ_AVX512_SOURCES}
        ${RABITQ_AVX512_POPCNT_SOURCES}
        ${RABITQ_HNSW_AVX2_SOURCES}
        ${RABITQ_HNSW_AVX512_CORE_SOURCES}
        ${RABITQ_HNSW_AVX512_POPCNT_SOURCES}
    )
    set(RABITQ_ARCH_DEFINITION RABITQ_TARGET_X86_64=1)
elseif(RABITQ_TARGET_PROCESSOR MATCHES "^(arm64|aarch64)$")
    list(APPEND RABITQ_ARCH_SOURCES ${RABITQ_NEON_SOURCES})
    set(RABITQ_ARCH_DEFINITION RABITQ_TARGET_AARCH64=1)
else()
    message(FATAL_ERROR
        "Unsupported RaBitQ target architecture '${RABITQ_TARGET_PROCESSOR}'; "
        "supported targets are x86_64/amd64 and arm64/aarch64"
    )
endif()

set(RABITQ_TEST_BACKEND_SOURCES)
if(RABITQ_BUILD_TESTS OR RABITQ_BUILD_BENCHMARKS)
    list(APPEND RABITQ_TEST_BACKEND_SOURCES
        src/simd/scalar_backend.cpp
        src/index/hnsw_search_scalar.cpp
    )
endif()

add_library(rabitq_core STATIC
    ${RABITQ_COMMON_SOURCES}
    ${RABITQ_ARCH_SOURCES}
    ${RABITQ_TEST_BACKEND_SOURCES}
)

target_include_directories(rabitq_core
    PUBLIC ${PROJECT_SOURCE_DIR}/include
    PRIVATE ${PROJECT_SOURCE_DIR}/src
)
target_compile_definitions(rabitq_core PUBLIC ${RABITQ_ARCH_DEFINITION})
target_compile_options(rabitq_core PRIVATE
    -Wall
    -Wextra
    $<$<CONFIG:Release>:-Ofast>
    $<$<CONFIG:Release>:-ftree-vectorize>
)
target_link_libraries(rabitq_core PUBLIC OpenMP::OpenMP_CXX)
set_target_properties(rabitq_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(RABITQ_BUILD_TESTS OR RABITQ_BUILD_BENCHMARKS)
    target_compile_definitions(rabitq_core PUBLIC RABITQ_ENABLE_TEST_BACKENDS=1)
endif()

if(RABITQ_TARGET_PROCESSOR MATCHES "^(x86_64|amd64)$")
    set_property(SOURCE ${RABITQ_AVX2_SOURCES} ${RABITQ_HNSW_AVX2_SOURCES}
        APPEND PROPERTY COMPILE_OPTIONS -mavx2 -mfma)
    set_property(SOURCE ${RABITQ_AVX512_SOURCES}
        APPEND PROPERTY COMPILE_OPTIONS -mavx512f -mavx512bw -mavx512dq -mfma)
    set_property(SOURCE ${RABITQ_AVX512_POPCNT_SOURCES} ${RABITQ_HNSW_AVX512_POPCNT_SOURCES}
        APPEND PROPERTY COMPILE_OPTIONS
        -mavx512f -mavx512bw -mavx512dq -mavx512vpopcntdq -mfma)
    set_property(SOURCE ${RABITQ_HNSW_AVX512_CORE_SOURCES}
        APPEND PROPERTY COMPILE_OPTIONS
        -mavx512f -mavx512bw -mavx512dq -mavx2 -mfma)
endif()

if(UNIX AND NOT APPLE)
    target_link_libraries(rabitq_core PUBLIC rt)
endif()

add_library(rabitq_headers INTERFACE)
target_include_directories(rabitq_headers INTERFACE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(rabitq_headers INTERFACE rabitq_core)

add_subdirectory(sample/cpp)

if(RABITQ_BUILD_PYTHON_BINDINGS)
    add_subdirectory(python_bindings)
endif()

# Testing
if(RABITQ_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(RABITQ_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()
