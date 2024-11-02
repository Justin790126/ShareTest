#include "ModelSktClnt.h"

ModelSktClnt::ModelSktClnt()
{

}

bool ModelSktClnt::connect()
{
    client_socket = socket(AF_INET, SOCK_STREAM, 0);
    char resMsg[1024];
    bool result = false;
    if (client_socket < 0) {
        sprintf(resMsg, "[ModelSktClnt] Error creating socket: %s", strerror(errno));
        m_sStatusMsg = std::move(resMsg);
        return result;
    }

    // Connect to the server
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = inet_addr(m_sIp.c_str()); // Replace with server IP
    server_addr.sin_port = htons(m_iPort); // Replace with server port

    if (::connect(client_socket, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        sprintf(resMsg, "[ModelSktClnt] Error connecting to server: %s", strerror(errno));
        m_sStatusMsg = std::move(resMsg);
        return result;
    }

    sprintf(resMsg, "[ModelSktClnt] Connected to server...");
        m_sStatusMsg = std::move(resMsg);
    result = true;
    return result;
}

void ModelSktClnt::Close()
{
    close(client_socket);
}