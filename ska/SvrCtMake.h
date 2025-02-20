#ifndef SVR_CT_MAKE_H
#define SVR_CT_MAKE_H

#include "RequestCtx.h"
#include <iostream>

using namespace std;

class SvrContourMake : public SvrState {
public:
    void handleRequest(ModelSktSvr* svr, PktRes* res) override;

    char* m_pcImg = NULL;
    float* m_pfImg = NULL;
    size_t m_sImgSize = 0;
    size_t m_sBatchSize = 4096*4;
    vector<pair<size_t, size_t>> m_vpPktOffsetAndPktSize;
};

#endif /* SVR_CT_MAKE_H */