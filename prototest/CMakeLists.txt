cmake_minimum_required(VERSION 3.5)
project(pb)

set(PROTOBUF_LIBRARY
    /usr/lib/libprotobuf.a)

include_directories(
    /usr/include    
    ${PROJECT_SOURCE_DIR}/include)

set(SOURCES_WRITTER
    writer.cpp
    ${PROJECT_SOURCE_DIR}/include/addressbook.pb.cc
)

add_executable(${PROJECT_NAME}_writter
    ${SOURCES_WRITTER})
target_link_libraries(${PROJECT_NAME}_writter ${PROTOBUF_LIBRARY})


set(SOURCES_READER
    reader.cpp
    ${PROJECT_SOURCE_DIR}/include/addressbook.pb.cc
)

add_executable(${PROJECT_NAME}_reader
    ${SOURCES_READER})
target_link_libraries(${PROJECT_NAME}_reader ${PROTOBUF_LIBRARY})

