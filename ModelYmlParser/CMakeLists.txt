cmake_minimum_required(VERSION 3.10)
project(yaml_parser)
set(CMAKE_CXX_STANDARD 11)

# Find the libyaml library
include_directories(/opt/homebrew/include/)

add_executable(${PROJECT_NAME} main.cpp)
target_link_directories(${PROJECT_NAME} PUBLIC /opt/homebrew/lib/)
target_link_libraries(${PROJECT_NAME} yaml-cpp)
