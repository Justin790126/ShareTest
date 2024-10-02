#include "ModelSktMsg.h"


ModelSktMsg::ModelSktMsg(/* args */)
{
}

ModelSktMsg::~ModelSktMsg()
{
}

char* ModelSktMsg::serializeInt(DType dtype, int data, size_t& outLen)
{
    int intDpktSize = sizeof(char) + sizeof(size_t) + sizeof(data);
    char* intPkt = new char[intDpktSize];
    memset(intPkt, 0, intDpktSize);

    int offset = 0;
    // dType (1bytes)
    char dataType = (char)dtype;
    memcpy(intPkt, &dataType, sizeof(dataType));
    offset+= sizeof(dataType);

    // data length in bytes (4bytes in integer/ 8 bytes)
    size_t sizeOfData = sizeof(data); // equal to 4
    memcpy(intPkt+offset, &sizeOfData, sizeof(size_t));
    offset += sizeof(size_t);

    // store integer data
    memcpy(intPkt+offset, &data, sizeof(int));
    offset += sizeof(int);

    outLen = offset;
    
    return intPkt;
}


char* ModelSktMsg::serializeFloat(DType dtype, float data, size_t& outLen)
{
    int intDpktSize = sizeof(char) + sizeof(size_t) + sizeof(data);
    char* intPkt = new char[intDpktSize];
    memset(intPkt, 0, intDpktSize);

    int offset = 0;
    // dType (1bytes)
    char dataType = (char)dtype;
    memcpy(intPkt, &dataType, sizeof(dataType));
    offset+= sizeof(dataType);

    // data length in bytes (4bytes in float/ 8 bytes)
    size_t sizeOfData = sizeof(data); // equal to 4
    memcpy(intPkt+offset, &sizeOfData, sizeof(size_t));
    offset += sizeof(size_t);

    // store float data
    memcpy(intPkt+offset, &data, sizeof(int));
    offset += sizeof(int);

    outLen = offset;
    
    return intPkt;
}

char* ModelSktMsg::serializeFloatArr(DType dtype, float* data, size_t dLen, size_t& outLen)
{
    int farrDpktSize = sizeof(char) + sizeof(size_t) + dLen*sizeof(float);
    char* farrPkt = new char[farrDpktSize];
    memset(farrPkt, 0, farrDpktSize);

    int offset = 0;
    // dType (1bytes)
    char dataType = (char)dtype;
    memcpy(farrPkt, &dataType, sizeof(dataType));
    offset+= sizeof(dataType);

    // data length in bytes (N bytes in integer/ 8 bytes)
    size_t sizeOfData = dLen*sizeof(float);
    memcpy(farrPkt+offset, &sizeOfData, sizeof(size_t));
    offset += sizeof(size_t);

    // store float array data
    memcpy(farrPkt+offset, data, dLen*sizeof(float));
    offset += dLen*sizeof(float);

    outLen = offset;
    
    return farrPkt;
}
