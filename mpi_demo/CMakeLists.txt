cmake_minimum_required(VERSION 3.10)
project(MPIExample)

# Find MPI package
find_package(MPI REQUIRED)

# Add the executable
add_executable(m1 m1.cpp)
add_executable(m2 m2.cpp)
add_executable(m3 m3.cpp)
add_executable(m4 m4.cpp)
add_executable(m5 m5.cpp)
add_executable(m6 m6.cpp)
add_executable(m7 m7.cpp)
add_executable(m8 m8.cpp)
add_executable(m9 m9.cpp)
# Link the MPI libraries to the executable
target_link_libraries(m1 MPI::MPI_CXX)
target_link_libraries(m2 MPI::MPI_CXX)
target_link_libraries(m3 MPI::MPI_CXX)
target_link_libraries(m4 MPI::MPI_CXX)
target_link_libraries(m5 MPI::MPI_CXX)
target_link_libraries(m6 MPI::MPI_CXX)
target_link_libraries(m7 MPI::MPI_CXX)
target_link_libraries(m8 MPI::MPI_CXX)
target_link_libraries(m9 MPI::MPI_CXX)