cmake_minimum_required(VERSION 2.8.12)
project(PolygonExample)

# Find the Qt4 package
find_package(Qt4 4.8 REQUIRED)

# Include the Qt4 macros
include(${QT_USE_FILE})

qt4_wrap_cpp(MOC_SOURCES
    MyWidget.h
    MainWidget.h
    )

# Add the executable
add_executable(painter
    main.cpp
    # ViewTableDialog.cpp
    # ViewColEdit.cpp
    ${MOC_SOURCES}
    )
# Link the Qt4 libraries to the executable
target_link_libraries(painter ${QT_LIBRARIES})

