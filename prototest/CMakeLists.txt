cmake_minimum_required(VERSION 3.5)
project(pb)
set(CMAKE_BUILD_TYPE "Debug")


set(PROTOBUF_LIBRARY
    /usr/lib/libprotobuf.a)

include_directories(
    /usr/include    
    ${PROJECT_SOURCE_DIR}/include)

# set(SOURCES_WRITTER
#     writer.cpp
#     ${PROJECT_SOURCE_DIR}/include/addressbook.pb.cc
# )

# add_executable(${PROJECT_NAME}_writter
#     ${SOURCES_WRITTER})
# target_link_libraries(${PROJECT_NAME}_writter ${PROTOBUF_LIBRARY})


# set(SOURCES_READER
#     reader.cpp
#     ${PROJECT_SOURCE_DIR}/include/addressbook.pb.cc
# )

# add_executable(${PROJECT_NAME}_reader
#     ${SOURCES_READER})
# target_link_libraries(${PROJECT_NAME}_reader ${PROTOBUF_LIBRARY})

include_directories(
    "/home/justin126/workspace/venv/lib64/python3.7/site-packages/tensorflow/include"
)

set(SOURCES_READER_TFE
    readtfe.cpp
)

add_executable(${PROJECT_NAME}_rtfe
    ${SOURCES_READER_TFE})
target_link_libraries(${PROJECT_NAME}_rtfe
    /home/justin126/tensorflow/bazel-bin/tensorflow/libtensorflow_framework.so
    /home/justin126/tensorflow/bazel-bin/tensorflow/libtensorflow_cc.so
    ${PROTOBUF_LIBRARY})


add_executable(${PROJECT_NAME}_rtfe1
readtfe1.cpp)
target_link_libraries(${PROJECT_NAME}_rtfe1
    /home/justin126/tensorflow/bazel-bin/tensorflow/libtensorflow_framework.so
    /home/justin126/tensorflow/bazel-bin/tensorflow/libtensorflow_cc.so
    ${PROTOBUF_LIBRARY})