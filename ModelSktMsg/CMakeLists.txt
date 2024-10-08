project(sktmsg)

set(CMAKE_CXX_STANDARD 11)

include_directories(/opt/homebrew/opt/openssl@1.1/include)

add_executable(skt
                main.cpp
                ModelSktMsg.cpp)
target_link_libraries(skt ssl crypto)

add_executable(skt_clnt
                socket_client.cpp
                ModelSktMsg.cpp)
target_link_libraries(skt_clnt ssl crypto)
add_executable(skt_svr
                socket_server.cpp
                ModelSktMsg.cpp)
target_link_libraries(skt_svr ssl crypto)
