cmake_minimum_required(VERSION 2.8.12)
project(treeTest)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find the Qt4 package
find_package(Qt4 4.8 REQUIRED QtCore QtGui)

# Include the Qt4 macros
include(${QT_USE_FILE})

qt4_wrap_cpp(MOC_SOURCES treeWidget.h)

# Add the executable
add_executable(${PROJECT_NAME}
    main.cpp
    treeWidget.cpp
    ${MOC_SOURCES}
    )

# Link the Qt4 libraries to the executable
target_link_libraries(${PROJECT_NAME} ${QT_LIBRARIES})