cmake_minimum_required(VERSION 3.14)
project(KalmanNoseTracker CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
message(STATUS "Build type : ${CMAKE_BUILD_TYPE}")

# ── OpenCV (needs contrib for cv::face::FacemarkLBF) ─────────────────────────
find_package(OpenCV REQUIRED
  COMPONENTS core imgproc highgui videoio objdetect face)

message(STATUS "OpenCV ${OpenCV_VERSION} @ ${OpenCV_INCLUDE_DIRS}")

# ── Executable ────────────────────────────────────────────────────────────────
add_executable(kalman_nose_tracker main.cpp)

target_include_directories(kalman_nose_tracker PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}   # pixel_kalman_tracker.hpp lives here
  ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(kalman_nose_tracker PRIVATE ${OpenCV_LIBS})

target_compile_options(kalman_nose_tracker PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:
    -Wall -Wextra
    $<$<CONFIG:Release>:-O3 -march=native>
  >
)