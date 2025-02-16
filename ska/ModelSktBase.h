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
#include <opencv2/opencv.hpp>
#include "ModelSktMsg.h"

using namespace std;

class ModelSktBase
{
    public:
        string GetStatusMsg() { return m_sStatusMsg; }
        bool Receive(vector<PktRes>& oRes);
        size_t BatchReceive(float* img);
        bool Send(char* pkt, size_t pktLen);
        bool Recv(char* buf, size_t pktLen);

        string m_sIp="127.0.0.1";
        int m_iPort=9527;
        int client_socket;
        string m_sStatusMsg;


    float* readPNGToFloat(const std::string& filePath, int width, int height);
    bool writeFloatToPNG(const std::string& outputPath, float* imageData, int width, int height);


    protected:
        char* m_pPkt=NULL;
    private:
        int m_iVerbose = 0;
};

#endif /* MODEL_SKT_BASE */