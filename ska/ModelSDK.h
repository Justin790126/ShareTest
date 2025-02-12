#ifndef MODEL_SDK_H
#define MODEL_SDK_H

#include <QThread>
#include <iostream>

#include "ModelSktMsg.h"
#include "ModelSktSvr.h"
#include "ModelSktClnt.h"

using namespace std;

class ModelSDK : public QThread
{
    Q_OBJECT
public:
    ModelSDK(QObject *parent = 0);
    ~ModelSDK() = default;

    void DlClose();
protected:
    virtual void run() override;

private:
    ModelSktMsg* m_svr = NULL;

    ModelSktClnt* m_clnt = NULL;
};

#endif /* MODEL_SDK_H */