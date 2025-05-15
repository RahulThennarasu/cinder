cmake_minimum_required(VERSION 3.10)
project(CompileML)

# Find LLVM
find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

# Include LLVM directories
include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Add subdirectories
add_subdirectory(src/compiler)

# Get proper shared library extension
if(APPLE)
  set(SHARED_LIB_EXT ".dylib")
else()
  set(SHARED_LIB_EXT ".so")
endif()

# Create a custom clang command that uses our pass
configure_file(
  ${CMAKE_SOURCE_DIR}/scripts/clang-ml.in
  ${CMAKE_BINARY_DIR}/bin/clang-ml
  @ONLY
)

# Make the script executable
execute_process(COMMAND chmod +x ${CMAKE_BINARY_DIR}/bin/clang-ml)

# Print configuration summary
message(STATUS "Configuration summary:")
message(STATUS "  LLVM version: ${LLVM_PACKAGE_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Install prefix: ${CMAKE_INSTALL_PREFIX}")
