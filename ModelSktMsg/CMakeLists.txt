project(sktmsg)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)


include_directories(/opt/homebrew/opt/openssl@1.1/include)

add_executable(skt
                main.cpp
                ModelSktMsg.cpp)
target_link_libraries(skt ssl crypto)

SET(SKT_SRC 
    ModelSktMsg.cpp
    ModelSktBase.cpp
    ModelSktSvr.cpp
    ModelSktClnt.cpp)

add_executable(skt_clnt
                socket_client.cpp
                ${SKT_SRC})
target_link_libraries(skt_clnt ssl crypto)
add_executable(skt_svr
                socket_server.cpp
                ${SKT_SRC})
target_link_libraries(skt_svr ssl crypto)
