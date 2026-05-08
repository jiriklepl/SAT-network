cmake_minimum_required(VERSION 3.20)

project(sat_network_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)

FetchContent_Declare(
  nlohmann_json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
  URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

FetchContent_Declare(
  cxxopts
  GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
  GIT_TAG v3.2.1
)

FetchContent_MakeAvailable(nlohmann_json cxxopts)

find_path(Z3_INCLUDE_DIR z3++.h)
find_library(Z3_LIBRARY z3 REQUIRED)

if(NOT Z3_INCLUDE_DIR OR NOT Z3_LIBRARY)
  message(FATAL_ERROR "Could not find Z3 headers/library")
endif()

add_executable(sat_synth_cpp
  cpp/cli.cpp
  cpp/config.cpp
  cpp/datasets.cpp
  cpp/encoding.cpp
  cpp/program.cpp
  cpp/sat_synth_cpp.cpp
)
target_include_directories(sat_synth_cpp PRIVATE ${Z3_INCLUDE_DIR})
target_link_libraries(sat_synth_cpp PRIVATE ${Z3_LIBRARY} cxxopts::cxxopts nlohmann_json::nlohmann_json)

include(CTest)
if(BUILD_TESTING)
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
