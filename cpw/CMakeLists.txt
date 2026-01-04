cmake_minimum_required(VERSION 3.5)
project(cpw)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 尋找 Qt 4.8
find_package(Qt4 4.8 REQUIRED QtCore QtGui)
include(${QT_USE_FILE})

# 1. 主程式部分 (MVC Viewer)
set(SOURCES 
    main.cpp 
    ModelCellProfile.cpp 
    ViewMainWindow.cpp
    lcMainWindow.cpp
)

set(HEADERS 
    ModelCellProfile.h 
    ViewMainWindow.h
    lcMainWindow.h 
)

# 執行 Qt MOC 處理
qt4_wrap_cpp(MOC_SRCS ${HEADERS})

add_executable(${PROJECT_NAME} ${SOURCES} ${MOC_SRCS})
target_link_libraries(${PROJECT_NAME} ${QT_LIBRARIES})

# 2. 獨立工具部分 (CellProfile Generator)
# 這是一個不依賴 Qt 的純 C++ 程式
add_executable(gen_prof gen_profile.cpp)