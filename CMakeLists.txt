cmake_minimum_required(VERSION 2.6)

project(HyperedgeCutting)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set (SRC
  Metric.cpp
  MetricUtils.cpp
  Metric_Hyperedge_Fixedcard.cpp
)

add_executable(testHyperEdgeUtil TestHyperEdgeUtil.cpp ${SRC})

set(CMAKE_CXX_FLAGS "-O3")        ## Optimize

