# CMakeLists.txt
cmake_minimum_required(VERSION 3.5)
project(xml)

find_package(Qt4 4.8 REQUIRED QtCore QtGui)
find_package(LibXml2 REQUIRED)
include(${QT_USE_FILE})
include_directories(${LIBXML2_INCLUDE_DIRS})

set(SOURCES
    main.cpp
    lcXml.cpp
    ViewXmlWafer.cpp
    ViewXmlTab.cpp
    ViewXmlItems.cpp
    ModelXmlParser.cpp
)

set(HEADERS
    lcXml.h
    ViewXmlWafer.h
    ViewXmlTab.h
    ViewXmlItems.h
    ModelXmlParser.h
)

qt4_wrap_cpp(SOURCES ${HEADERS})


add_executable(${PROJECT_NAME}
    ${SOURCES})

target_link_libraries(${PROJECT_NAME} ${QT_LIBRARIES} ${LIBXML2_LIBRARIES})