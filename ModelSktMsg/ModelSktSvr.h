#ifndef MODEL_SKT_SVR
#define MODEL_SKT_SVR

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
#include "ModelSktBase.h"

using namespace std;

class ModelSktSvr : public ModelSktBase
{
    public:
        ModelSktSvr();
        ~ModelSktSvr() = default;
        bool init();
        void start();
        bool Accept();
        void Close();

    private:
        int server_socket;
        struct sockaddr_in server_addr, client_addr;
        socklen_t client_addr_len;

        bool m_bSvrStop = false;
    
    private:
        int DlClose(u_char syncFlg=0x00);
        char* m_pDlClosePkt=NULL;

        void ContourMake(u_char flg=0x00);
        float* m_pfCurContour=NULL;
        double m_dCurContourIdx=0;
        double m_dContourPktId = 0;
        int m_iBatchId = 0;


};

#endif /* MODEL_SKT_SVR */