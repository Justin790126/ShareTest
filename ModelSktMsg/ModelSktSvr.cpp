#include "ModelSktSvr.h"

ModelSktSvr::ModelSktSvr()
{

}

bool ModelSktSvr::init()
{
    bool result = false;
    char resMsg[1024];
    server_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (server_socket < 0) {
        sprintf(resMsg, "[ModelSktSvr] Error creating socket: %s", strerror(errno));
        m_sStatusMsg = std::move(resMsg);
        return result;
    }

    // Bind the socket to a port
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = inet_addr(m_sIp.c_str());
    server_addr.sin_port = htons(8080); // Replace with desired port

    if (::bind(server_socket, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        sprintf(resMsg, "[ModelSktSvr] Error binding socket: %s", strerror(errno));
        m_sStatusMsg = std::move(resMsg);
        return result;
    }

    // Listen for incoming connections
    if (listen(server_socket, 5) < 0) {
        sprintf(resMsg, "[ModelSktSvr] Error listening on socket: %s", strerror(errno));
        m_sStatusMsg = std::move(resMsg);
        return result;
    }

    sprintf(resMsg, "[ModelSktSvr] Server listening on port 8080...");
    m_sStatusMsg = std::move(resMsg);
    result = false;
    return result;
}

bool ModelSktSvr::Accept()
{
    char resMsg[1024];
    client_addr_len = sizeof(client_addr);
    client_socket = accept(server_socket, (struct sockaddr*)&client_addr, &client_addr_len);
    if (client_socket < 0) {
        sprintf(resMsg, "[ModelSktSvr] Error accepting connection: %s", strerror(errno)); 
        m_sStatusMsg = std::move(resMsg);
        return false;
    }
    return true;
}

void ModelSktSvr::start()
{
    while (!m_bSvrStop) {
        client_addr_len = sizeof(client_addr);
        if (!Accept()) {
            m_bSvrStop = true;
            break;
        }

        vector<PktRes> res;
        if (!Receive(res)) {
            m_bSvrStop = true;
            break;
        }

        if (res.empty()) {
            m_bSvrStop = true;
            m_sStatusMsg = "[ModelSktSvr] Parsing Result is empty";
            break;
        }

        // Getstatusmsg
        #if TEST_ALL
        ModelSktMsg msg;
        size_t msgSize = m_sStatusMsg.length()+1;
        size_t respktlen;
        char* response = new char[msgSize];
        strcpy(response, m_sStatusMsg.c_str());
        msg.serializeArr<char>(response, msgSize, respktlen);
        char* respkt = msg.createPkt(respktlen);
        Send(respkt, respktlen);
        #endif

        close(client_socket);
    }

    if (m_bSvrStop) {
        // send response to client and close
        close(client_socket);
    }
    
}

void ModelSktSvr::Close()
{
    close(server_socket);
}