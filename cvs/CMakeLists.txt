cmake_minimum_required(VERSION 3.5)
project(cvs)

find_package(Qt4 4.8 REQUIRED QtCore QtGui)
include(${QT_USE_FILE})

set(SOURCES
    main.cpp
    MainWindow.cpp
    LayoutCanvas.cpp
)

set(HEADERS
    MainWindow.h
    LayoutCanvas.h
)

qt4_wrap_cpp(MOC_SOURCES ${HEADERS})

add_executable(${PROJECT_NAME}
    ${SOURCES}
    ${MOC_SOURCES}
)

target_link_libraries(${PROJECT_NAME} ${QT_LIBRARIES})
