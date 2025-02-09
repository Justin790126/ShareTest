#ifndef MODEL_SKT_BASE
#define MODEL_SKT_BASE

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <iostream>
#include <vector>
#include "ModelSktMsg.h"

using namespace std;

class ModelSktBase
{
    public:
        string GetStatusMsg() { return m_sStatusMsg; }
        bool Receive(vector<PktRes>& oRes);
        size_t BatchReceive(float* img);
        void Send(char* pkt, size_t pktLen);

        string m_sIp="127.0.0.1";
        int m_iPort=9527;
        int client_socket;
        string m_sStatusMsg;

    protected:
        char* m_pPkt=NULL;
        ushort m_usVerbose = 0;
    private:
};

#endif /* MODEL_SKT_BASE */