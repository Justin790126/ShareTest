# CMakeLists.txt
cmake_minimum_required(VERSION 3.5)
project(cw)
include_directories(/opt/homebrew/include/yaml-cpp)
# Check for nullptr support

find_package(Qt5 COMPONENTS Core Gui Widgets REQUIRED)

find_package(Qt5 COMPONENTS PrintSupport REQUIRED)

set(SOURCES
    main.cpp
    ChartWizard.cpp
    ViewChartWizard.cpp
    Section.cpp
    qcustomplot.cpp
)

set(HEADERS
    ChartWizard.h
    ViewChartWizard.h
    Section.h
    qcustomplot.h
)

qt5_wrap_cpp(SOURCES ${HEADERS})

add_executable(${PROJECT_NAME}
    ${SOURCES}
    )
target_link_directories(${PROJECT_NAME} PUBLIC /opt/homebrew/lib/)
target_link_libraries(${PROJECT_NAME} Qt5::Core Qt5::Gui Qt5::Widgets Qt5::PrintSupport yaml-cpp)

