add_executable(rabitq_simd_benchmark simd_benchmark.cpp)
target_include_directories(rabitq_simd_benchmark PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(rabitq_simd_benchmark PRIVATE rabitq_headers)
target_compile_options(rabitq_simd_benchmark PRIVATE
    -Wall
    -Wextra
    $<$<CONFIG:Release>:-Ofast>
)
