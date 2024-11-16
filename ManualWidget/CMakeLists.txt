# CMakeLists.txt
cmake_minimum_required(VERSION 3.5)
project(manual)
include_directories(/opt/homebrew/include/yaml-cpp)
# Check for nullptr support

set(MD4C_SRC md4c.cpp)
set(MD4C_HTML md2html.cpp cmdline.cpp md4c-html.cpp entity.cpp )

find_package(Qt5 COMPONENTS Core Gui Widgets REQUIRED)

find_package(Qt5 COMPONENTS PrintSupport REQUIRED)

set(SOURCES
    main.cpp
    ViewManual.cpp
)

set(HEADERS
ViewManual.h
)

qt5_wrap_cpp(SOURCES ${HEADERS})



add_executable(${PROJECT_NAME}
    ${SOURCES}
    ${MD4C_SRC}
    ${MD4C_HTML})
    target_link_directories(${PROJECT_NAME} PUBLIC /opt/homebrew/lib/)
target_link_libraries(${PROJECT_NAME} Qt5::Core Qt5::Gui Qt5::Widgets Qt5::PrintSupport yaml-cpp)





# cmake_minimum_required(VERSION 3.10)
# project(yaml_parser)
# set(CMAKE_CXX_STANDARD 11)

# # Find the libyaml library
# 

# add_executable(${PROJECT_NAME} main.cpp)
# target_link_directories(${PROJECT_NAME} PUBLIC /opt/homebrew/lib/)
# target_link_libraries(${PROJECT_NAME} yaml-cpp)
