cmake_minimum_required(VERSION 3.11)
project(HPGT VERSION 0.1.0)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Debug)
ENDIF()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

if(CMAKE_C_COMPILER_ID STREQUAL "GNU" OR CMAKE_C_COMPILER_ID STREQUAL "Clang")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -O3")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3")
  set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -march=native")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")
elseif(CMAKE_C_COMPILER_ID STREQUAL "MSVC")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /W3 /O2")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3 /O2")
endif()

include(CheckCXXCompilerFlag)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  CHECK_CXX_COMPILER_FLAG("-std=c++17" COMPILER_SUPPORTS_CXX17)
  if(COMPILER_SUPPORTS_CXX17)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
    message(STATUS "Using flag -std=c++17.")
  else()
    message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++17 support. Please use a different C++ compiler.")
  endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 19.10)
    message(FATAL_ERROR "The MSVC compiler version ${CMAKE_CXX_COMPILER_VERSION} does not support C++17. Please use MSVC 2017 or later.")
  else()
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    add_compile_options(/bigobj)
    message(STATUS "MSVC compiler supports C++17.")
  endif()
endif()

set(LIBRARY_NAME HPGT)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

find_package(nlohmann_json REQUIRED)
message(STATUS "nlohmann_json version: ${nlohmann_json_VERSION}")

find_package(spdlog REQUIRED)
message(STATUS "spdlog version: ${spdlog_VERSION}")

find_package(Eigen3 REQUIRED)
message(STATUS "Eigen3 version: ${Eigen3_VERSION}")

find_package(Sophus)
message(STATUS "Sophus version: ${Sophus_VERSION}")

find_package(Ceres REQUIRED)
message(STATUS "Ceres version: ${CERES_VERSION}")

file(GLOB_RECURSE SOURCES "src/*.cc")
add_library(${LIBRARY_NAME} SHARED ${SOURCES})

target_include_directories(
${LIBRARY_NAME} PUBLIC
${PROJECT_SOURCE_DIR}/include
${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(
${LIBRARY_NAME} PUBLIC
nlohmann_json::nlohmann_json
spdlog::spdlog_header_only
Sophus::Sophus
Ceres::ceres
)

add_executable(RunHPGT app/run_hpgt.cc)
target_link_libraries(RunHPGT ${LIBRARY_NAME})

add_executable(RunCorrelationToff app/run_correlation_toff.cc)
target_link_libraries(RunCorrelationToff ${LIBRARY_NAME})

file(GLOB_RECURSE TEST_SOURCES  "module_test/*.cc")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin/module_test)
foreach(TEST_SOURCE ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    target_link_libraries(${TEST_NAME} ${LIBRARY_NAME})
endforeach()
