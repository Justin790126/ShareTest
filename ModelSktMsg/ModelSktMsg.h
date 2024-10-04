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
            pkt_len (8bytes), chksum (32bytes), sender(1bytes), response(1bytes), number of parameters (4bytes), dtype(1bytes), data_len(8bytes), data , data_end_byte...

store          size_t              char[32]          char            char                 int                        char         size_t                     char   
 */

class ModelSktMsg
{

public:
    ModelSktMsg(/* args */);
    ~ModelSktMsg();

    bool verifyChksum(u_char* clnt, u_char* svr);

    template <typename T>
    char* serialize(DType dtype, T data, size_t& outLen);
    
    template <typename T>
    char* serializeArr(DType dtype, T* data, size_t dLen, size_t& outLen);

    std::vector<std::pair<char*, size_t>>* GetDataSections() { return &m_vDataSection; }
    void clearDataSection();
    void generateChecksum(char* data, size_t sizeOfData, u_char* chksum);

    char* createPkt(size_t& outLen);

    template<typename T>
    T deserialize(const char* data);

    template<typename T>
    void deserializeArr(T* out, char* pkt, size_t numOfBytes);

    void printPkt(char* pkg, size_t dsize);

    int getDataSectionOffset();
private:
    /* Test usage */
    char* serializeInt(DType dtype, int data, size_t& outLen);
    char* serializeFloat(DType dtype, float data, size_t& outLen);
    char* serializeFloatArr(DType dtype, float* data, size_t dLen, size_t& outLen);

    // <data, pkt size>
    std::vector<std::pair<char*,size_t>> m_vDataSection;

};

#endif /* MODEL_SKT_MSG */