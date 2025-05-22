# CMakeLists.txt
cmake_minimum_required(VERSION 3.5)
project(itp)

# Check for nullptr support


find_package(Qt4 4.8 REQUIRED QtCore QtGui)
include(${QT_USE_FILE})

set(SOURCES
    main.cpp
    mainwindow.cpp
    qcustomplot.cpp
)

set(HEADERS
    mainwindow.h
    qcustomplot.h
)

qt4_wrap_cpp(SOURCES ${HEADERS})


add_executable(${PROJECT_NAME}
    ${SOURCES})

target_link_libraries(${PROJECT_NAME} ${QT_LIBRARIES})





# cmake_minimum_required(VERSION 3.10)
# project(yaml_parser)
# set(CMAKE_CXX_STANDARD 11)

# # Find the libyaml library
# 

# add_executable(${PROJECT_NAME} main.cpp)
# target_link_directories(${PROJECT_NAME} PUBLIC /opt/homebrew/lib/)
# target_link_libraries(${PROJECT_NAME} yaml-cpp)
