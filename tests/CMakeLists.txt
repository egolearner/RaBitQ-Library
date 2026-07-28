cmake_minimum_required(VERSION 3.10)

# Prefer an installed Google Test; CMake 3.10 has no FetchContent fallback.
find_package(GTest QUIET)
if(GTest_FOUND OR GTEST_FOUND)
    if(TARGET GTest::gtest)
        set(RABITQ_GTEST_LIBRARIES GTest::gtest GTest::gtest_main)
    elseif(TARGET GTest::GTest)
        set(RABITQ_GTEST_LIBRARIES GTest::GTest GTest::Main)
    else()
        set(RABITQ_GTEST_LIBRARIES ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES})
    endif()
else()
    if(CMAKE_VERSION VERSION_LESS 3.11)
        message(FATAL_ERROR
            "Building RaBitQ tests with CMake 3.10 requires an installed Google Test"
        )
    endif()

    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    if(CMAKE_VERSION VERSION_LESS 3.14)
        FetchContent_GetProperties(googletest)
        if(NOT googletest_POPULATED)
            FetchContent_Populate(googletest)
            add_subdirectory(
                ${googletest_SOURCE_DIR}
                ${googletest_BINARY_DIR}
                EXCLUDE_FROM_ALL
            )
        endif()
    else()
        FetchContent_MakeAvailable(googletest)
    endif()
    set(RABITQ_GTEST_LIBRARIES gtest gtest_main)
endif()

# Automatically register all tests under unit and integration folders
file(GLOB_RECURSE UNIT_TESTS ${CMAKE_CURRENT_SOURCE_DIR}/unit/*_test.cpp)
file(GLOB_RECURSE INTEG_TESTS ${CMAKE_CURRENT_SOURCE_DIR}/integration/*_test.cpp)
file(GLOB_RECURSE COMMON_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/common/*.cpp)

# add executables
add_executable(rabitq_tests
    ${COMMON_SRCS}
    ${UNIT_TESTS}
    ${INTEG_TESTS}
)

# Include directories
target_include_directories(rabitq_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${PROJECT_SOURCE_DIR}/src
    ${GTEST_INCLUDE_DIRS}
)
# Compiler flags
target_compile_options(rabitq_tests PRIVATE -Wall -Wextra)
# Link google test and rabitqlib headers
target_link_libraries(rabitq_tests
    PRIVATE
    ${RABITQ_GTEST_LIBRARIES}
    rabitq_headers
)

# Discover tests for CTest
include(GoogleTest)
gtest_discover_tests(rabitq_tests)
