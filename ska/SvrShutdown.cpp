#include "SvrShutdown.h"


void SvrShutdown::handleRequest(ModelSktSvr* svr, PktRes* res)
{
    ModelSktMsg resMsg;
    string test = "bye!";
    char* testmsg = new char[sizeof(test) + 1];
    strcpy(testmsg, test.c_str());
    size_t resLen;
    resMsg.serializeArr<char>(testmsg, sizeof(test) + 1, resLen);
    char* resmsgpkt = resMsg.createPkt(resLen);
    svr->Send(resmsgpkt, resLen);
    svr->SetSvrStop(true);
}