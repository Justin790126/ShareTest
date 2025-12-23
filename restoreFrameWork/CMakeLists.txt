cmake_minimum_required(VERSION 3.10)

# Project name and language
project(QtUndoExample VERSION 1.0 LANGUAGES CXX)

# Find the Qt4 package
find_package(Qt4 REQUIRED QtCore QtGui)

# Include Qt macros
include(${QT_USE_FILE})

# Enable automatic Meta-Object Compiler processing
set(CMAKE_AUTOMOC ON)

# Add the executable target
add_executable(QtUndoExample main.cpp)

# Link the target with Qt libraries
target_link_libraries(QtUndoExample ${QT_LIBRARIES})