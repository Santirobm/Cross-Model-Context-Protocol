cmake_minimum_required(VERSION 3.16)
project(cmcp VERSION 0.1.0 LANGUAGES C)

option(CMCP_BUILD_TESTS "Build CMCP C99 unit tests" ON)
option(CMCP_BUILD_LLAMA_HARNESS "Build C11 llama.cpp benchmark harness" OFF)
set(CMCP_LLAMA_DIR "" CACHE PATH "Path to llama.cpp root directory")

set(CMCP_CORE_SOURCES
    core/validation/cmcp_crc32c.c
    core/protocol/cmcp_header.c
    core/serialization/cmcp_tlv.c
    core/validation/cmcp_compat.c
    core/transport/cmcp_shm_channel.c
)

function(cmcp_apply_strict_c99 target_name)
    set_target_properties(${target_name} PROPERTIES
        C_STANDARD 99
        C_STANDARD_REQUIRED ON
        C_EXTENSIONS OFF
    )
    if(MSVC)
        target_compile_options(${target_name} PRIVATE /W4 /WX)
    else()
        target_compile_options(${target_name} PRIVATE -Wall -Wextra -Werror -pedantic)
    endif()
endfunction()

# Static library
add_library(cmcp STATIC ${CMCP_CORE_SOURCES})
target_include_directories(cmcp PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/core/include>
    $<INSTALL_INTERFACE:include>
)
cmcp_apply_strict_c99(cmcp)

# Shared library
add_library(cmcp_shared SHARED ${CMCP_CORE_SOURCES})
target_include_directories(cmcp_shared PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/core/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_definitions(cmcp_shared
    PUBLIC CMCP_SHARED
    PRIVATE CMCP_BUILD_SHARED
)
set_target_properties(cmcp_shared PROPERTIES OUTPUT_NAME "cmcp")
if(WIN32)
    set_target_properties(cmcp_shared PROPERTIES PREFIX "")
endif()
cmcp_apply_strict_c99(cmcp_shared)

# C99 unit tests
if(CMCP_BUILD_TESTS)
    enable_testing()
    add_executable(test_cmcp_core tests/c/test_cmcp_core.c)
    target_link_libraries(test_cmcp_core PRIVATE cmcp)
    cmcp_apply_strict_c99(test_cmcp_core)
    add_test(NAME test_cmcp_core COMMAND test_cmcp_core)
endif()

# Optional C11 llama.cpp benchmark & kv-compat harness
if(CMCP_BUILD_LLAMA_HARNESS)
    if(NOT CMCP_LLAMA_DIR)
        message(FATAL_ERROR "CMCP_LLAMA_DIR must be set when CMCP_BUILD_LLAMA_HARNESS=ON")
    endif()

    find_library(LLAMA_IMPORT_LIB
        NAMES llama libllama.dll.a libllama
        PATHS "${CMCP_LLAMA_DIR}/build/src" "${CMCP_LLAMA_DIR}/build/bin" "${CMCP_LLAMA_DIR}/build"
        NO_DEFAULT_PATH
    )
    if(NOT LLAMA_IMPORT_LIB)
        message(FATAL_ERROR "Could not find libllama in ${CMCP_LLAMA_DIR}/build")
    endif()

    add_executable(cmcp_llama_harness benchmarks/cmcp_llama_harness.c)
    set_target_properties(cmcp_llama_harness PROPERTIES
        C_STANDARD 11
        C_STANDARD_REQUIRED ON
        C_EXTENSIONS OFF
    )
    target_include_directories(cmcp_llama_harness PRIVATE
        "${CMCP_LLAMA_DIR}/include"
        "${CMCP_LLAMA_DIR}/ggml/include"
    )
    target_link_libraries(cmcp_llama_harness PRIVATE cmcp "${LLAMA_IMPORT_LIB}")
    if(WIN32)
        target_link_libraries(cmcp_llama_harness PRIVATE psapi)
    else()
        target_link_libraries(cmcp_llama_harness PRIVATE m)
    endif()
    if(MSVC)
        target_compile_options(cmcp_llama_harness PRIVATE /W4 /WX)
    else()
        target_compile_options(cmcp_llama_harness PRIVATE -Wall -Wextra -Werror -pedantic)
    endif()
endif()
