#ifndef MODEL_SKT_MSG
#define MODEL_SKT_MSG

#include <iostream>
#include <cstdlib>

enum DType
{
    DTYPE_INT=0x09,
    DTYPE_FLOAT,
    DTYPE_FLOAT_ARR
};

/*
            pkt_len (8bytes), chksum (32bytes), sender(1bytes), response(1bytes), number of parameters (4bytes), dtype(1bytes), data_len(8bytes), data ...

store          size_t              char[32]          char            char                 int                        char         size_t        
 */

class ModelSktMsg
{

public:
    ModelSktMsg(/* args */);
    ~ModelSktMsg();

    char* serializeInt(DType dtype, int data, size_t& outLen);
    char* serializeFloat(DType dtype, float data, size_t& outLen);
    char* serializeFloatArr(DType dtype, float* data, size_t dLen, size_t& outLen);



private:
    char* m_pDataSection=NULL;

};


#endif /* MODEL_SKT_MSG */