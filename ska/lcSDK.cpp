
#include "lcSDK.h"

lcSDK::lcSDK(QObject *parent): QObject(parent)
{
    model = new ModelSDK();

    m_clnt = new ModelSktClnt;
    view = new TestWidget();
    model->start();
    connect(view->btnSend, SIGNAL(clicked()), this, SLOT(handleSendMsg()));
    connect(view->btnSendContourMaked, SIGNAL(clicked()), this, SLOT(handleSenContourMakedMsg()));
    view->show();
}

void lcSDK::handleSenContourMakedMsg()
{
    model->ContourMake();
}


void lcSDK::handleSendMsg()
{
    cout << "send msg" << endl;
    // ModelSktMsg msg;
    // size_t pktLen;
    model->DlClose();
    // msg.serialize<char>(0x00, pktLen);
    // char* pkt = msg.createPkt(pktLen, SVR_DLCLOSE, 0x00, (char)flg);
    // m_clnt->Send(pkt, pktLen);



}