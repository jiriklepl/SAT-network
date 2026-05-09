cmake_minimum_required(VERSION 3.20)

project(sat_network_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)

FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.12.0
)

FetchContent_Declare(
  cxxopts
  GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
  GIT_TAG v3.3.1
)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.14.0
)

FetchContent_MakeAvailable(nlohmann_json cxxopts Catch2)

find_path(Z3_INCLUDE_DIR z3++.h)
find_library(Z3_LIBRARY z3 REQUIRED)

if(NOT Z3_INCLUDE_DIR OR NOT Z3_LIBRARY)
  message(FATAL_ERROR "Could not find Z3 headers/library")
endif()

add_library(sat_synth_core
  cpp/assumptions.cpp
  cpp/cli.cpp
  cpp/config.cpp
  cpp/datasets.cpp
  cpp/encoding.cpp
  cpp/program.cpp
  cpp/solver.cpp
)
target_include_directories(sat_synth_core PUBLIC cpp ${Z3_INCLUDE_DIR})
target_link_libraries(sat_synth_core PUBLIC ${Z3_LIBRARY} cxxopts::cxxopts nlohmann_json::nlohmann_json)

add_executable(sat_synth_cpp cpp/sat_synth_cpp.cpp)
target_link_libraries(sat_synth_cpp PRIVATE sat_synth_core)

include(CTest)
if(BUILD_TESTING)
  add_executable(cpp_unit_tests cpp/tests/cpp_unit_tests.cpp)
  target_sources(cpp_unit_tests PRIVATE
    cpp/tests/assumptions_tests.cpp
    cpp/tests/cli_tests.cpp
    cpp/tests/config_tests.cpp
    cpp/tests/datasets_tests.cpp
    cpp/tests/program_tests.cpp
    cpp/tests/solver_tests.cpp
  )
  target_link_libraries(cpp_unit_tests PRIVATE sat_synth_core Catch2::Catch2WithMain)
  add_test(NAME cpp_unit_tests COMMAND cpp_unit_tests)

  find_package(Python3 COMPONENTS Interpreter)
  set(SAT_NETWORK_TEST_PYTHON "" CACHE FILEPATH "Python interpreter for integration tests")
  if(NOT SAT_NETWORK_TEST_PYTHON AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.venv/bin/python")
    set(SAT_NETWORK_TEST_PYTHON "${CMAKE_CURRENT_SOURCE_DIR}/.venv/bin/python")
  elseif(NOT SAT_NETWORK_TEST_PYTHON)
    set(SAT_NETWORK_TEST_PYTHON ${Python3_EXECUTABLE})
  endif()
  if(SAT_NETWORK_TEST_PYTHON)
    add_test(
      NAME cpp_solver_integration
      COMMAND
        ${CMAKE_COMMAND} -E env
        SAT_SYNTH_CPP=$<TARGET_FILE:sat_synth_cpp>
        ${SAT_NETWORK_TEST_PYTHON} -m unittest test_cpp_solver
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
  endif()
endif()
