# CMakeLists.txt
cmake_minimum_required(VERSION 3.5)
project(doc)

# Check for nullptr support

set(MD4C_SRC md4c.cpp)
set(MD4C_HTML md2html.cpp cmdline.cpp md4c-html.cpp entity.cpp )

find_package(Qt4 4.8 REQUIRED QtCore QtGui)
include(${QT_USE_FILE})

set(SOURCES
    main.cpp
    ViewManual.cpp
    lcDoc.cpp
    ModelMdReader.cpp
)

set(HEADERS
ViewManual.h
ModelMdReader.h
lcDoc.h
)

qt4_wrap_cpp(SOURCES ${HEADERS})

add_executable(${PROJECT_NAME}
    ${SOURCES}
    ${MD4C_SRC}
    ${MD4C_HTML})
target_link_libraries(${PROJECT_NAME} ${QT_LIBRARIES})





# cmake_minimum_required(VERSION 3.10)
# project(yaml_parser)
# set(CMAKE_CXX_STANDARD 11)

# # Find the libyaml library
# 

# add_executable(${PROJECT_NAME} main.cpp)
# target_link_directories(${PROJECT_NAME} PUBLIC /opt/homebrew/lib/)
# target_link_libraries(${PROJECT_NAME} yaml-cpp)
