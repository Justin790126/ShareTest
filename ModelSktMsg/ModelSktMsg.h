#ifndef MODEL_SKT_MSG
#define MODEL_SKT_MSG

#include <iostream>
#include <cstdlib>
#include <vector>
#include <openssl/sha.h>

enum DType
{
    DTYPE_INT=0x09,
    DTYPE_FLOAT,
    DTYPE_DOUBLE,
    DTYPE_INT_ARR,
    DTYPE_FLOAT_ARR,
    DTYPE_DOUBLE_ARR,
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

    template <typename T>
    char* serialize(DType dtype, T data, size_t& outLen);
    char* serializeInt(DType dtype, int data, size_t& outLen);
    char* serializeFloat(DType dtype, float data, size_t& outLen);

    template <typename T>
    char* serializeArr(DType dtype, T* data, size_t dLen, size_t& outLen);
    char* serializeFloatArr(DType dtype, float* data, size_t dLen, size_t& outLen);

    std::vector<std::pair<char*, size_t>>* GetDataSections() { return &m_vDataSection; }
    void ClearDataSection();
    void generateChecksum(char* data, size_t sizeOfData, u_char* chksum);

    char* createPkt(size_t& outLen);

    void printPkt(char* pkg, size_t dsize);
private:
    // <data, pkt size>
    std::vector<std::pair<char*,size_t>> m_vDataSection;

};


#endif /* MODEL_SKT_MSG */