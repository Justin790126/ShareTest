#ifndef REQUEST_CTX_H
#define REQUEST_CTX_H

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "ModelSktMsg.h"
#include "ModelSktSvr.h"

using namespace std;

class SvrState;
class SvrShutdown;

class SvrState
{
protected:
    // add sdk share data for derived classes
public:
    virtual void handleRequest(ModelSktSvr* svr, PktRes* res) = 0;
    virtual ~SvrState() = default;
};


class RequestHandlerContext {
private:
    SvrState* m_svrState;

public:
    void setState(SvrState* newState) {
        m_svrState = newState;
    }

    void handleRequest(ModelSktSvr* svr, PktRes* res) {
        if (m_svrState) {
            m_svrState->handleRequest(svr, res);
        }
    }

    ~RequestHandlerContext() {
        delete m_svrState;
    }
};

#endif /* REQUEST_CTX_H */