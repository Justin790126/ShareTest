#include <iostream>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <string.h>
#include "ModelSktMsg.h"
#include "ModelSktClnt.h"

using namespace std;

ModelSktClnt clnt;

void DlClose(u_char flg)
{
    ModelSktMsg msg;
    size_t pktLen;
    msg.serialize<char>(0x00, pktLen);
    char* pktSt = msg.createPkt(pktLen, SVR_DLCLOSE, 0x00, (char)flg);
    clnt.Send(pktSt, pktLen);

    vector<PktRes> response;
    clnt.Receive(response);

    if (response.size() != 1) {
        printf("[socket_client/DlClose]: response pkt wrong :%zu\n", response.size());
        return;
    }
    PktRes resFrmSvr = response[0];
    char* resMsg = (char*)resFrmSvr.arr;
    printf("[socket_client/DlClose]: %s\n", resMsg);


    if (pktSt) delete[] pktSt;

    
    clnt.Close();
}

void ContourMake()
{

}

int main() {
    // int client_socket;
    // struct sockaddr_in server_addr;
    // if (!clnt.connect()) {
    //     printf("%s\n", clnt.GetStatusMsg().c_str());
    // }
    
    // // Send a message to the server

    
#if TEST_ALL
    ModelSktMsg msg;
    string path = "aasdfghjkjhgfdsasdfgh.qmdl";
    size_t sizePath = strlen(path.c_str())+1;
    char* cpath = new char[sizePath];
    strcpy(cpath, path.c_str());

    size_t pktLen;
    msg.serialize<int>(999, pktLen);
    msg.serializeArr<char>(cpath, sizePath, pktLen);
    msg.serialize<float>(1.234, pktLen);
    float fatest[5] = {1.1, 2.2, 3.3, 4.4, 5.5};
    char *farrPkt = msg.serializeArr<float>(fatest, 5, pktLen);
    msg.serialize<double>(3.141592654321, pktLen);
    double dbtest[5] = {1.1111111111, 2.222222222, 3.33333333, 4.333334, 5.2345678765435};
    msg.serializeArr<double>(dbtest, 5, pktLen);
    int ittest[5] = {1, 3, 5, 7, 9};
    msg.serializeArr<int>(ittest, 5, pktLen);
    char* pkt = msg.createPkt(pktLen);

    // char* message = "Hello from the client!";
    printf("[socket_client] pktLen = %zu \n", pktLen);
    msg.printPkt(pkt, pktLen);
    clnt.Send(pkt, pktLen);

    vector<PktRes> res;
    clnt.Receive(res);

    char* resMsg = (char*)res[0].arr;
    printf("[socket_client] res = %s\n", resMsg);
#endif

    // DlClose();
    int c;
    printf("Enter key\n");
    while (true) {
        
        c = getchar();
        printf("Enter key %c\n", c);
        
        switch (c)
        {
        case 'c':
            if (!clnt.connect()) {
                printf("%s\n", clnt.GetStatusMsg().c_str());
                break;
            }
            DlClose(0x00);
            break;
        case 'f':
            if (!clnt.connect()) {
                printf("%s\n", clnt.GetStatusMsg().c_str());
                break;
            }
            DlClose(0x01);
            break;
        case 'q':
            break;
        default:
            break;
        }
    }


    clnt.Close();
    cout << "eof client" << endl;
    
    return 0;
}