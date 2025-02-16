#include "ModelSDK.h"
#include <QApplication>
ModelSDK::ModelSDK(QObject *parent)
    : QThread(parent)
{
    if (!m_clnt) {
        m_clnt = new ModelSktClnt;
    }
}
void ModelSDK::ContourMake()
{
    cout << "contour make" << endl;
    m_clnt->connect();
    ModelSktMsg msg;
    size_t pktLen;
    msg.serialize<char>(0x00, pktLen);
    char* pkt = msg.createPkt(pktLen, SVR_CONTOUR_MAKE, 0x01, 0x00, 0x00);
    m_clnt->Send(pkt, pktLen);
    bool echo = false;

    // char rcv[1024];
    // m_clnt->Recv(rcv, sizeof(rcv));
    // msg.printPkt(rcv, sizeof(rcv));

    size_t imgSize = 0;
    size_t imgBatchSize = 0;
    size_t numOfReqs = 0;
    while (!echo) {
        vector<PktRes> res;
        m_clnt->Receive(res);
        // cout << res.size() << endl;
        if (res.size() > 0) {
            imgSize = res[0].sData;
            imgBatchSize = res[1].sData;
            numOfReqs = res[2].sData;
            echo = true;
        }
        usleep(1000);
        QApplication::processEvents();
    }
    cout << "Start to receive packets: " << imgSize << endl;
    cout << "batch size: " << imgBatchSize << endl;
    cout << "Number of requests: " << numOfReqs << endl;

    float* data = new float[imgSize];
    memset(data, 0, imgSize);
    size_t recvSize = 0;
    m_clnt->connect();
    // for (size_t i =0;i< numOfReqs; i++) {
    for (size_t i =0;i< numOfReqs; i++) {
        
            ModelSktMsg msg;
            size_t pktLen;
            msg.serialize<char>(0x00, pktLen);
            char* pkt = msg.createPkt(pktLen, SVR_CONTOUR_MAKE, 0x01, 0x01, i);
        m_clnt->Send(pkt, pktLen);

            size_t bytesToRecv = imgBatchSize;
            if (recvSize + bytesToRecv > imgSize) {
                bytesToRecv = imgSize - recvSize;
            }
            char *buf = new char[bytesToRecv];
        m_clnt->Recv(buf, bytesToRecv);
            // // copy buf to data
            memcpy(data+recvSize, buf, bytesToRecv);
            if (buf) delete [] buf;
            printf("id : %zu, offset : %zu, size : %zu \n", i, recvSize, bytesToRecv);

            // // msg.printPkt(data+recvSize, bytesToRecv);
            recvSize += bytesToRecv;
        
        
    }
    m_clnt->Close();
    m_clnt->writeFloatToPNG("modelsdk.png", data, 4096, 4096);

    if (data) delete[] data;
    m_clnt->Close();
}

void ModelSDK::DlClose()
{
    m_clnt->connect();
    string test = "Hello, world!";
    char* testmsg = new char[sizeof(test) + 1];
    strcpy(testmsg, test.c_str());

    ModelSktMsg msg;
    size_t pktLen;
    msg.serializeArr<char>(testmsg, sizeof(test) + 1, pktLen);
    
    char* pkt = msg.createPkt(pktLen, SVR_SHUTDOWN, 0x0a, (char)0x0b, 0x0C);
    // msg.printPkt(pkt, pktLen);
    m_clnt->Send(pkt, pktLen);
    // delete[] pkt;
    vector<PktRes> res;
    m_clnt->Receive(res);
    m_clnt->Close();
}

void ModelSDK::run()
{
    pid_t child_pid = fork();
    cout << "child_pid: " << child_pid << endl;
    if (child_pid <0) {
        cout << "Failed to fork" << endl;
        return;
    } else if (child_pid == 0) {
        ModelSktSvr svr;
        svr.init();
        svr.start();
        svr.Close();
        cout << "EOF Child process: " << child_pid << endl;
        exit(0);
    } else {
        waitpid(child_pid,NULL,0);

        while(true);
    }
    cout << "EOF SDK svr: " << child_pid << endl;
}