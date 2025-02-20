#ifndef SVR_SHUTDOWN_H
#define SVR_SHUTDOWN_H

#include "RequestCtx.h"

#include <iostream>

using namespace std;

class SvrShutdown : public SvrState {
public:
    void handleRequest(ModelSktSvr* svr, PktRes* res) override;
};

#endif /* SVR_SHUTDOWN_H */