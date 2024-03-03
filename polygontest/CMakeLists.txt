# CMakeLists.txt
cmake_minimum_required(VERSION 3.5)
project(MovablePolygon)

# Check for nullptr support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)

if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
    add_definitions(-Dnullptr=0)
endif()


find_package(Qt5 COMPONENTS Core Gui Widgets REQUIRED)

set(SOURCES
    main.cpp
    hexagonitem.cpp
    mainwindow.cpp
)

set(HEADERS
    hexagonitem.h
    mainwindow.h
)


qt5_wrap_cpp(SOURCES ${HEADERS})



add_executable(${PROJECT_NAME}
    ${SOURCES})

target_link_libraries(${PROJECT_NAME} Qt5::Core Qt5::Gui Qt5::Widgets)
