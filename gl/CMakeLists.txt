cmake_minimum_required(VERSION 2.8)

project(OpenGLTest)

# Find Qt4
find_package(Qt4 REQUIRED COMPONENTS QtCore QtGui QtOpenGL)
include(${QT_USE_FILE})

# Find OpenGL
find_package(OpenGL REQUIRED)

# Define the source files
set(SOURCES
    main.cpp
)

# Create the executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Link libraries
target_link_libraries(${PROJECT_NAME}
    ${QT_LIBRARIES}
    ${OPENGL_LIBRARIES}
)

# Ensure Qt4 definitions are included
add_definitions(${QT_DEFINITIONS})