cmake_minimum_required(VERSION 3.12)
project(OpenMPTest LANGUAGES CXX)
set(CMAKE_BUILD_TYPE Debug)
# Find OpenMP
find_package(OpenMP REQUIRED)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)

# Add the executable
# add_executable(OpenMPTest main.cpp)

# add_executable(filter main_filter.cpp)
add_executable(m41 m41.cpp)
add_executable(m42 m42.cpp)
add_executable(m43 m43.cpp)
add_executable(tree1 tree1.cpp)
add_executable(tree2 tree2.cpp)
# Link against OpenMP
# target_link_libraries(OpenMPTest PUBLIC OpenMP::OpenMP_CXX)
# target_link_libraries(filter PUBLIC OpenMP::OpenMP_CXX)
target_link_libraries(m41 PUBLIC OpenMP::OpenMP_CXX)
target_link_libraries(m42 PUBLIC OpenMP::OpenMP_CXX)
target_link_libraries(m43 PUBLIC OpenMP::OpenMP_CXX)
target_link_libraries(tree1 PUBLIC OpenMP::OpenMP_CXX)
target_link_libraries(tree2 PUBLIC OpenMP::OpenMP_CXX)


