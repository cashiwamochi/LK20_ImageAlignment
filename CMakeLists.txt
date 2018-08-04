cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(LK20_Tracker)

IF (!APPLE)
  set(CMAKE_CXX_COMPILER g++)
ENDIF()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2")

# OpenCV
# You can set path to OpenCVConfig.cmake
set(OpenCV_DIR ~/.local/OpenCV/share/OpenCV)
find_package(OpenCV REQUIRED)
IF (OpenCV_FOUND)
  message([message] OpenCV_VER : ${OpenCV_VERSION})
  include_directories(${OpenCV_INCLUDE_DIRS})
  link_directories(${OpenCV_LIBRARY_DIRS})
  add_definitions(${OpenCV_DEFINITIONS})
ENDIF()

# Ceres Solver
# http://ceres-solver.org/
option(WITH_CERES "USE CERES" OFF)
find_package(Ceres)
IF (CERES_FOUND)
  option(WITH_CERES "USE CERES" ON)
  message([message] CERES_VER : ${CERES_VERSION})
  include_directories(${CERES_INCLUDE_DIRS})
  link_directories(${CERES_LIBRARY_DIRS})
  add_definitions(${CERES_DEFINITIONS})
ENDIF()

if(WITH_CERES)
  add_definitions(-DENABLE_CERES)
endif()

include_directories(${PROJECT_SOURCE_DIR}/src)

add_executable(
  LK20_Tracker
  LK_example.cc
  ./src/Simulator.cpp
  ./src/LK20.cpp
  )

target_link_libraries(
  LK20_Tracker
  ${OpenCV_LIBS}
  ${CERES_LIBRARIES}  
  )
