#ifndef MODEL_SKT_MSG
#define MODEL_SKT_MSG

#include <iostream>
#include <cstdlib>
#include <vector>
#include <string>
#include <openssl/sha.h>

using namespace std;

enum SvrCmd
{
    SVR_DLCLOSE = 0x01,
    SVR_DLOPEN,
    SVR_SETLYT,
    SVR_CONTOUR_MAKE
};

enum DType
{
    DTYPE_INT=0x11,
    DTYPE_FLOAT,
    DTYPE_DOUBLE,
    DTYPE_CHAR,
    DTYPE_SIZE_T,
    DTYPE_INT_ARR,
    DTYPE_FLOAT_ARR,
    DTYPE_DOUBLE_ARR,
    DTYPE_CHAR_ARR
};

enum SyncFlag
{
    SYNC_START,
    SYNC_INPROGRESS,
    SYNC_END
};

class PktRes
{
    public:
        DType dType;
        char cSender;
        char cResCode;
        char cSyncFlg;
        int pktId;

        void* arr;
        size_t arrSize;

        int iData;
        double dData;
        float fData;
        char cData;
        size_t sData;
};

/*
            pkt_len (8bytes), chksum (32bytes), sender(1bytes), response(1bytes), sync_flag, pkt_id, number of parameters (4bytes), dtype(1bytes), data_len(8bytes), data , data_end_byte...

store          size_t              char[32]          char            char             char       int          int                        char         size_t                     char
 */

class ModelSktMsg
{

public:
    ModelSktMsg(/* args */);
    ~ModelSktMsg();

    bool verifyChksum(u_char *clnt, u_char *svr);

    size_t getChksumSize() {return (size_t)SHA256_DIGEST_LENGTH;}

    template <typename T>
    char *serialize(T data, size_t &outLen);

    template <typename T>
    char *serializeArr(T *data, size_t dLen, size_t &outLen);

    std::vector<std::pair<char *, size_t>> *GetDataSections() { return &m_vDataSection; }
    void clearDataSection();
    void generateChecksum(char *data, size_t sizeOfData, u_char *chksum);

    char *createPkt(size_t &outLen, char sender=0x00, char response=0x00, char syncFlag=0x00, int pktId=0);

    template <typename T>
    T deserialize(const char *data);

    template <typename T>
    void deserializeArr(T *out, char *pkt, size_t numOfBytes);

    void printPkt(char *pkg, size_t dsize);

    int getDataSectionOffset();

private:
    // <data, pkt size>
    std::vector<std::pair<char *, size_t>> m_vDataSection;
    char endByte = 0xAB;
};

#endif /* MODEL_SKT_MSG */