cmake_minimum_required(VERSION 3.5)
project(cpw)
set(CMAKE_CXX_STANDARD 11)

find_package(Qt4 4.8 REQUIRED QtCore QtGui)
include(${QT_USE_FILE})

set(SOURCES main.cpp ModelCellProfile.cpp ViewMainWindow.cpp lcMainWindow.cpp)
set(HEADERS ModelCellProfile.h ViewMainWindow.h lcMainWindow.h)
qt4_wrap_cpp(MOC_SRCS ${HEADERS})

add_executable(${PROJECT_NAME} ${SOURCES} ${MOC_SRCS})
target_link_libraries(${PROJECT_NAME} ${QT_LIBRARIES})
add_executable(gen_prof gen_profile.cpp)