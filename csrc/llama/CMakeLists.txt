# CMakeLists.txt for llama.cpp gene conversion integration.
#
# Usage:
#   mkdir build && cd build
#   cmake .. -DLLAMA_CPP_DIR=/path/to/llama.cpp
#   make -j$(nproc)
#
# This produces libmolly_llama.so (or .dll) that can be loaded
# via ctypes from Python (see llama_bridge.py).

cmake_minimum_required(VERSION 3.16)
project(molly_llama LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ── llama.cpp dependency ────────────────────────────────────────────────────
# Set LLAMA_CPP_DIR to the root of your llama.cpp checkout.
# We link against the built llama library.

if(NOT DEFINED LLAMA_CPP_DIR)
    set(LLAMA_CPP_DIR "${CMAKE_SOURCE_DIR}/../../third_party/llama.cpp"
        CACHE PATH "Path to llama.cpp source root")
endif()

if(EXISTS "${LLAMA_CPP_DIR}/include/llama.h")
    set(LLAMA_INCLUDE_DIR "${LLAMA_CPP_DIR}/include")
elseif(EXISTS "${LLAMA_CPP_DIR}/llama.h")
    set(LLAMA_INCLUDE_DIR "${LLAMA_CPP_DIR}")
else()
    message(WARNING "llama.cpp not found at ${LLAMA_CPP_DIR}. "
                    "Set -DLLAMA_CPP_DIR=/path/to/llama.cpp")
endif()

# ── Sources ─────────────────────────────────────────────────────────────────

add_library(molly_llama SHARED
    genome_gguf.cpp
    llama_gene_convert.cpp
)

target_include_directories(molly_llama PRIVATE
    ${CMAKE_SOURCE_DIR}/..     # csrc/utils.h
    ${LLAMA_INCLUDE_DIR}
)

# Link against llama.cpp library
find_library(LLAMA_LIB NAMES llama
    PATHS "${LLAMA_CPP_DIR}/build" "${LLAMA_CPP_DIR}/build/src"
    NO_DEFAULT_PATH)

if(LLAMA_LIB)
    target_link_libraries(molly_llama PRIVATE ${LLAMA_LIB})
else()
    message(WARNING "libllama not found. Build llama.cpp first, then re-run cmake.")
endif()

# Install
install(TARGETS molly_llama LIBRARY DESTINATION lib)
