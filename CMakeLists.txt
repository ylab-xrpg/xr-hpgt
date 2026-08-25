cmake_minimum_required(VERSION 3.16)

project(HPGT VERSION 0.1.0 LANGUAGES CXX)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release
                                               RelWithDebInfo MinSizeRel)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

find_package(nlohmann_json REQUIRED)
message(STATUS "nlohmann_json version: ${nlohmann_json_VERSION}")

find_package(spdlog REQUIRED)
message(STATUS "spdlog version: ${spdlog_VERSION}")

find_package(Eigen3 REQUIRED)
message(STATUS "Eigen3 version: ${Eigen3_VERSION}")

find_package(Sophus REQUIRED)
message(STATUS "Sophus version: ${Sophus_VERSION}")

# Sophus uses an external fmt package when one is available, while spdlog uses
# its bundled fmt by default. Reuse Sophus's fmt for spdlog headers to avoid
# mixing incompatible fmt versions in the same translation unit. If no external
# fmt target exists, retain spdlog's bundled backend.
add_library(hpgt_spdlog INTERFACE)
target_link_libraries(hpgt_spdlog INTERFACE spdlog::spdlog_header_only)
if(TARGET fmt::fmt)
  target_compile_definitions(hpgt_spdlog INTERFACE SPDLOG_FMT_EXTERNAL)
  target_link_libraries(hpgt_spdlog INTERFACE fmt::fmt)
  message(STATUS "spdlog fmt backend: external fmt shared with Sophus")
else()
  message(STATUS "spdlog fmt backend: bundled")
endif()
add_library(HPGT::spdlog ALIAS hpgt_spdlog)

find_package(Ceres REQUIRED)
message(STATUS "Ceres version: ${CERES_VERSION}")

set(HPGT_SOURCES
    src/config/system_config.cc
    src/estimator/estimator.cc
    src/estimator/factor_adder.cc
    src/initializer/i2p_extrinsic_initializer.cc
    src/initializer/imu_preintegrator.cc
    src/initializer/initializer_helper.cc
    src/initializer/p2p_extrinsic_initializer.cc
    src/initializer/spatial_extrinsic_initializer.cc
    src/initializer/time_offset_initializer.cc
    src/sensor_data/imu_data.cc
    src/sensor_data/pose_data.cc
    src/sensor_data/sensor_data_manager.cc)

add_library(HPGT SHARED ${HPGT_SOURCES})

target_compile_features(HPGT PUBLIC cxx_std_17)
target_include_directories(HPGT PUBLIC "${PROJECT_SOURCE_DIR}/include")

target_link_libraries(
  HPGT
  PUBLIC Eigen3::Eigen
         nlohmann_json::nlohmann_json
         HPGT::spdlog
         Sophus::Sophus
         Ceres::ceres)

function(hpgt_enable_warnings TARGET_NAME)
  if(MSVC)
    target_compile_options(${TARGET_NAME} PRIVATE /W3 /bigobj)
  else()
    target_compile_options(${TARGET_NAME} PRIVATE -Wall)
  endif()
endfunction()

hpgt_enable_warnings(HPGT)

if(MSVC)
  set_target_properties(HPGT PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

add_executable(RunHPGT app/run_hpgt.cc)
target_link_libraries(RunHPGT PRIVATE HPGT)
hpgt_enable_warnings(RunHPGT)

add_executable(RunCorrelationToff app/run_correlation_toff.cc)
target_link_libraries(RunCorrelationToff PRIVATE HPGT)
hpgt_enable_warnings(RunCorrelationToff)

include(CTest)

if(BUILD_TESTING)
  set(HPGT_TEST_DATA_DIR "${PROJECT_SOURCE_DIR}/data/simulated")
  set(HPGT_TEST_OUTPUT_DIR "${CMAKE_BINARY_DIR}/test_output")

  set(HPGT_TEST_TARGETS
      initializer_test
      sensor_data_manager_test
      spline_test
      system_config_test)

  foreach(TEST_TARGET IN LISTS HPGT_TEST_TARGETS)
    add_executable(${TEST_TARGET} "tests/${TEST_TARGET}.cc")
    target_link_libraries(${TEST_TARGET} PRIVATE HPGT)
    hpgt_enable_warnings(${TEST_TARGET})
    set_target_properties(
      ${TEST_TARGET}
      PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")
  endforeach()

  add_test(
    NAME system_config_test
    COMMAND system_config_test "${HPGT_TEST_DATA_DIR}"
            "${HPGT_TEST_OUTPUT_DIR}/system_config")
  add_test(
    NAME sensor_data_manager_test
    COMMAND sensor_data_manager_test "${HPGT_TEST_DATA_DIR}"
            "${HPGT_TEST_OUTPUT_DIR}/sensor_data_manager")
  add_test(NAME initializer_test
           COMMAND initializer_test "${HPGT_TEST_DATA_DIR}")
  add_test(
    NAME spline_test
    COMMAND spline_test "${HPGT_TEST_DATA_DIR}"
            "${HPGT_TEST_OUTPUT_DIR}/spline")
endif()
