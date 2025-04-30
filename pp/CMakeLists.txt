# CMakeLists.txt
cmake_minimum_required(VERSION 3.5)
project(ra)

# Check for nullptr support


find_package(Qt4 4.8 REQUIRED QtCore QtGui)
include(${QT_USE_FILE})

set(SOURCES1
    main.cpp
    ra.cpp
    ModelTreeGen.cpp
)
set(HEADERS1
    ra.h
    ModelTreeGen.h
)
qt4_wrap_cpp(SOURCES1 ${HEADERS1})
add_executable(${PROJECT_NAME}1
    ${SOURCES1})
target_link_libraries(${PROJECT_NAME}1 ${QT_LIBRARIES})


set(SOURCES2
    stra.cpp
    ra.cpp
    ModelTreeGen.cpp
)
set(HEADERS2
    ra.h
    ModelTreeGen.h
)
qt4_wrap_cpp(SOURCES2 ${HEADERS2})
add_executable(${PROJECT_NAME}2
    ${SOURCES2})
target_link_libraries(${PROJECT_NAME}2 ${QT_LIBRARIES})



set(SOURCES3
    mtra.cpp
    ra.cpp
    ModelTreeGen.cpp
)
set(HEADERS3
    ra.h
    ModelTreeGen.h
)
qt4_wrap_cpp(SOURCES3 ${HEADERS3})
add_executable(${PROJECT_NAME}3
    ${SOURCES3})
target_link_libraries(${PROJECT_NAME}3 ${QT_LIBRARIES})



set(SOURCES4
    mtra2.cpp
    ra.cpp
    ModelTreeGen.cpp
)
set(HEADERS4
    ra.h
    ModelTreeGen.h
)
qt4_wrap_cpp(SOURCES4 ${HEADERS4})
add_executable(${PROJECT_NAME}4
    ${SOURCES4})
target_link_libraries(${PROJECT_NAME}4 ${QT_LIBRARIES})



set(SOURCES5
    wid.cpp
    ra.cpp
    ModelTreeGen.cpp
)
set(HEADERS5
    ra.h
    ModelTreeGen.h
)
qt4_wrap_cpp(SOURCES5 ${HEADERS5})
add_executable(${PROJECT_NAME}5
    ${SOURCES5})
target_link_libraries(${PROJECT_NAME}5 ${QT_LIBRARIES})



set(SOURCES6
    it.cpp
    ra.cpp
    ModelTreeGen.cpp
)
set(HEADERS6
    ra.h
    ModelTreeGen.h
)
qt4_wrap_cpp(SOURCES6 ${HEADERS6})
add_executable(${PROJECT_NAME}6
    ${SOURCES5})
target_link_libraries(${PROJECT_NAME}6 ${QT_LIBRARIES})
# cmake_minimum_required(VERSION 3.10)
# project(yaml_parser)
# set(CMAKE_CXX_STANDARD 11)

# # Find the libyaml library
# 

# add_executable(${PROJECT_NAME} main.cpp)
# target_link_directories(${PROJECT_NAME} PUBLIC /opt/homebrew/lib/)
# target_link_libraries(${PROJECT_NAME} yaml-cpp)
