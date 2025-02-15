#include "ModelSDK.h"

ModelSDK::ModelSDK(QObject *parent)
    : QThread(parent)
{
    if (!m_clnt) {
        m_clnt = new ModelSktClnt;
    }
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
    cout << "total pkt" << endl;
    msg.printPkt(pkt, pktLen);
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