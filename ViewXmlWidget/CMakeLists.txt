# CMakeLists.txt
cmake_minimum_required(VERSION 3.5)
project(xml)

find_package(Qt4 4.8 REQUIRED QtCore QtGui)
include(${QT_USE_FILE})

set(SOURCES
    main.cpp
    lcXml.cpp
    ViewXmlWafer.cpp
    ViewXmlTab.cpp
    ModelXmlParser.cpp
)

set(HEADERS
    lcXml.h
    ViewXmlWafer.h
    ViewXmlTab.h
    ModelXmlParser.h
)

qt4_wrap_cpp(SOURCES ${HEADERS})


add_executable(${PROJECT_NAME}
    ${SOURCES})

target_link_libraries(${PROJECT_NAME} ${QT_LIBRARIES})
