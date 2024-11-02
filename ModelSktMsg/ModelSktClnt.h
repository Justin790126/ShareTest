#ifndef MODEL_SKT_CLNT
#define MODEL_SKT_CLNT

#include <iostream>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <string.h>
#include "ModelSktBase.h"

using namespace std;

class ModelSktClnt : public ModelSktBase
{
    public:
        ModelSktClnt();
        ~ModelSktClnt() = default;

        bool connect();
        void Send(char* pkt, size_t pktLen);
        void Close();

    private:
        int client_socket;
        struct sockaddr_in server_addr;
};

#endif /* MODEL_SKT_CLNT */