#ifndef REQUEST_CTX_H
#define REQUEST_CTX_H

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "ModelSktMsg.h"

class State {
public:
    virtual void handleRequest(ModelSktMsg* resFromClnt) = 0;
    virtual ~State() = default;
};

// class StateSyncFlg0 : public State {
// public:
//     void handleRequest(ModelSktMsg* resFromClnt) override {
//         // Handle the case for cSyncFlg == 0x00 (Initial request)
//         m_pfImg = readPNGToFloat("lena4096_4096.png", 4096, 4096);
//         writeFloatToPNG("lenaqq.png", m_pfImg, 4096, 4096);
//         // Further image handling, memory allocation, and batch sending logic
//     }
// };

// class StateSyncFlg1 : public State {
// public:
//     void handleRequest(ModelSktMsg* resFromClnt) override {
//         // Handle the case for cSyncFlg == 0x01 (Batch sending request)
//         int id = resFromClnt->pktId;
//         auto pair = m_vpPktOffsetAndPktSize[id];
//         size_t offset = pair.first;
//         size_t size = pair.second;
//         // Copy and send batch data
//     }
// };

// class StateSyncFlg2 : public State {
// public:
//     void handleRequest(ModelSktMsg* resFromClnt) override {
//         // Handle the case for cSyncFlg == 0x02 (Cleanup request)
//         if (m_pfImg) delete[] m_pfImg;
//         m_pfImg = NULL;
//         if (m_pcImg) delete[] m_pcImg;
//         m_pcImg = NULL;
//         // Send response to client that cleanup is done
//     }
// };

// class RequestHandlerContext {
// private:
//     State* state;

// public:
//     void setState(State* newState) {
//         state = newState;
//     }

//     void handleRequest(ModelSktMsg* resFromClnt) {
//         if (state) {
//             state->handleRequest(resFromClnt);
//         }
//     }

//     ~RequestHandlerContext() {
//         delete state;
//     }
// };

// void processRequest(ModelSktMsg* resFromClnt) {
//     RequestHandlerContext context;

//     // Set the state based on the value of cSyncFlg
//     switch (resFromClnt->cSyncFlg) {
//         case 0x00:
//             context.setState(new StateSyncFlg0());
//             break;
//         case 0x01:
//             context.setState(new StateSyncFlg1());
//             break;
//         case 0x02:
//             context.setState(new StateSyncFlg2());
//             break;
//         default:
//             std::cerr << "Unknown sync flag!" << std::endl;
//             return;
//     }

//     // Delegate the request handling to the current state
//     context.handleRequest(resFromClnt);
// }


#endif /* REQUEST_CTX_H */