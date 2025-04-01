# CMakeLists.txt
cmake_minimum_required(VERSION 3.5)
project(cg)

# set(CMAKE_CXX_STANDARD 11)

set(SOURCES
    main.cpp
)

add_executable(${PROJECT_NAME}
    ${SOURCES})

add_library(${PROJECT_NAME}lib
    Vector.cpp
    GeoUtils.cpp
    Line.h
    Plane.h)
