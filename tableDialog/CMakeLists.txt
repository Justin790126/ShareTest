cmake_minimum_required(VERSION 2.8.12)
project(PolygonExample)

# Find the Qt4 package
find_package(Qt4 4.8 REQUIRED)

# Include the Qt4 macros
include(${QT_USE_FILE})

qt4_wrap_cpp(MOC_SOURCES
    ViewTableDialog.h
    ViewColEdit.h
    )

# Add the executable
add_executable(l4
    main.cpp
    ViewTableDialog.cpp
    ViewColEdit.cpp
    ${MOC_SOURCES}
    )
# Link the Qt4 libraries to the executable
target_link_libraries(l4 ${QT_LIBRARIES})


qt4_wrap_cpp(MOC_SOURCES2
    CustomTableWidget.h
    )

add_executable(tbl
    table.cpp
    CustomTableWidget.cpp
    ${MOC_SOURCES2}
)
target_link_libraries(tbl ${QT_LIBRARIES})

add_executable(treeSetting
treeSetting.cpp
)
target_link_libraries(treeSetting ${QT_LIBRARIES})

add_executable(toolbtns
toolbtns.cpp
)
target_link_libraries(toolbtns ${QT_LIBRARIES})


add_executable(gaugeTree
gaugeTreeWidget.cpp
)
target_link_libraries(gaugeTree ${QT_LIBRARIES})

