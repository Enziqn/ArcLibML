cmake_minimum_required(VERSION 3.16)
project(ArcML VERSION 0.1.0 LANGUAGES CXX)

# Header-only library target
add_library(ArcML INTERFACE)
add_library(ArcML::ArcML ALIAS ArcML)

target_include_directories(ArcML INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_compile_features(ArcML INTERFACE cxx_std_17)

option(ARCML_BUILD_EXAMPLES "Build examples" ON)
option(ARCML_BUILD_TESTS    "Build tests" ON)

if(ARCML_BUILD_EXAMPLES)
  add_executable(arcml_dense_demo examples/dense_demo.cpp)
  target_link_libraries(arcml_dense_demo PRIVATE ArcML::ArcML)
endif()

if(ARCML_BUILD_TESTS)
  enable_testing()
  add_executable(arcml_dense_test tests/dense_test.cpp)
  target_link_libraries(arcml_dense_test PRIVATE ArcML::ArcML)
  add_test(NAME dense_test COMMAND arcml_dense_test)
endif()