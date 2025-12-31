cmake_minimum_required(VERSION 3.10)
project(ProfileTreeViewerQt4 CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Qt4 REQUIRED)
include(${QT_USE_FILE})
add_definitions(${QT_DEFINITIONS})

set(SRCS
  main.cpp
  MainWindow.cpp
  ViewProfTreeItem.cpp
  ModelProfTree.cpp
)

set(HDRS
  MainWindow.h
  ViewProfTreeItem.h
  ModelProfTree.h
)

# Qt4 需要 moc
qt4_wrap_cpp(MOC_SRCS MainWindow.h ModelProfTree.h ViewProfTreeItem.h)

add_executable(ProfileTreeViewer
  ${SRCS}
  ${HDRS}
  ${MOC_SRCS}
)

target_link_libraries(ProfileTreeViewer ${QT_LIBRARIES})
