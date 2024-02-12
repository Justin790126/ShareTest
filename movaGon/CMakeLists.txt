cmake_minimum_required(VERSION 2.8.12)
project(PolygonExample)

# Find the Qt4 package
find_package(Qt4 4.8 REQUIRED QtCore QtGui)

# Include the Qt4 macros
include(${QT_USE_FILE})

#qt4_wrap_cpp(MOC_SOURCES MyRect.h Bullet.h)

# Add the executable
add_executable(l4
    main.cpp
    )

# Link the Qt4 libraries to the executable
target_link_libraries(l4 ${QT_LIBRARIES})