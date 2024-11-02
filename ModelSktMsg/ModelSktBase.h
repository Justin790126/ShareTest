#ifndef MODEL_SKT_BASE
#define MODEL_SKT_BASE

#include "ModelSktMsg.h"
#include <iostream>

using namespace std;

class ModelSktBase
{
    public:
        string GetStatusMsg() { return m_sStatusMsg; }
        string m_sIp="127.0.0.1";
        int m_iPort=8080;
        string m_sStatusMsg;
};

#endif /* MODEL_SKT_BASE */