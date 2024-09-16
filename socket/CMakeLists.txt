cmake_minimum_required(VERSION 3.13)
project(socket_example)

add_executable(socket_server socket_server.cpp)
add_executable(socket_client socket_client.cpp)


find_package(Qt5Core REQUIRED)
find_package(Qt5Widgets REQUIRED)

add_executable(qt_example main.cpp)
target_link_libraries(qt_example Qt5::Core Qt5::Widgets)