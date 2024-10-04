project(sktmsg)

set(CMAKE_CXX_STANDARD 11)

include_directories(/opt/homebrew/opt/openssl@1.1/include)

add_executable(skt
                main.cpp
                ModelSktMsg.cpp)
target_link_libraries(skt ssl crypto)
