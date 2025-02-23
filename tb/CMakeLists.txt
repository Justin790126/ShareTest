# CMakeLists.txt
cmake_minimum_required(VERSION 3.5)
project(tb)

find_package(Qt4 4.8 REQUIRED QtCore QtGui)
include(${QT_USE_FILE})


include_directories(/usr/local/include)
include_directories(/usr/local/include/cryptopp)

set(SOURCES
    ende.cpp
    ModelCrypto.cpp
)

set(HEADERS
    ModelCrypto.h
)

qt4_wrap_cpp(SOURCES ${HEADERS})


add_executable(${PROJECT_NAME}_ende
    ${SOURCES})

target_link_libraries(${PROJECT_NAME}_ende ${QT_LIBRARIES} /usr/local/lib/libcryptopp.a)


set(PROTOBUF_LIBRARY
    /usr/lib/libprotobuf.a)
set(TF_INCLUDE_DIR
    "/home/justin126/workspace/venv/lib64/python3.7/site-packages/tensorflow/include")
set(TF_LIBRARY
    /home/justin126/tensorflow/bazel-bin/tensorflow/libtensorflow_framework.so
    /home/justin126/tensorflow/bazel-bin/tensorflow/libtensorflow_cc.so)
include_directories(
    /usr/include
    ${TF_INCLUDE_DIR}
) 

set(SOURCES1
    main.cpp
    lcTensorBoard.cpp
    ViewTensorBoard.cpp
    ViewTimeSeries.cpp
    qcustomplot.cpp
    PropsSection.cpp
    ModelTfParser.cpp
    utils.cpp
)

set(HEADERS1
    lcTensorBoard.h
    ViewTensorBoard.h
    ViewTimeSeries.h
    qcustomplot.h
    PropsSection.h
    ModelTfParser.h
    utils.h
)

qt4_wrap_cpp(SOURCES1 ${HEADERS1})


add_executable(${PROJECT_NAME}
    ${SOURCES1})

target_link_libraries(${PROJECT_NAME}
    ${QT_LIBRARIES}
    /usr/local/lib/libcryptopp.a
    ${TF_LIBRARY}
    ${PROTOBUF_LIBRARY}
    )


