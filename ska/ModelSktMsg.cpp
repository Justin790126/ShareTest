#include "ModelSktMsg.h"

ModelSktMsg::ModelSktMsg(/* args */)
{
    m_vDataSection.clear();
}

void ModelSktMsg::clearDataSection()
{
    for (int i = 0; i < m_vDataSection.size(); i++)
    {
        if (m_vDataSection[i].first)
        {
            delete[] m_vDataSection[i].first;
        }
    }
    m_vDataSection.clear();
}

ModelSktMsg::~ModelSktMsg()
{
    clearDataSection();
}

void ModelSktMsg::printPkt(char *pkg, size_t dsize)
{
    for (size_t i = 0; i < dsize; i++)
    {
        printf("%02x ", (u_char)pkg[i]);
    }
    printf("\n");
}

void ModelSktMsg::generateChecksum(char *data, size_t sizeOfData, u_char *chksum)
{
    if (!m_bGenCksum)
        return;
    SHA256(reinterpret_cast<const unsigned char *>(data), sizeOfData, chksum);
}

int ModelSktMsg::getDataSectionOffset()
{
    int chksumLen = m_bGenCksum ? SHA256_DIGEST_LENGTH : 0;
    return sizeof(size_t) + chksumLen + sizeof(char) + sizeof(char) + sizeof(char) + sizeof(int) + sizeof(int);
}

char *ModelSktMsg::createPkt(size_t &outLen, char sender, char response, char syncFlag, int pktId)
{
    if (m_vDataSection.empty())
        return NULL;

    size_t totalSizeOfDataSection = 0;
    for (int i = 0; i < m_vDataSection.size(); i++)
    {
        totalSizeOfDataSection += m_vDataSection[i].second;
    }

    char *dataSection = new char[totalSizeOfDataSection];
    memset(dataSection, 0, totalSizeOfDataSection);
    int offset = 0;
    for (int i = 0; i < m_vDataSection.size(); i++)
    {
        char *data = m_vDataSection[i].first;
        memmove(dataSection + offset, data, m_vDataSection[i].second);
        offset += m_vDataSection[i].second;
    }

    // printf("----total data section : %zu----\n", totalSizeOfDataSection);
    // printPkt(dataSection, offset);
    char *chksum = NULL;
    size_t chkSumSize = 0;
    if (m_bGenCksum)
    {
        chkSumSize = SHA256_DIGEST_LENGTH;
        u_char checksum[SHA256_DIGEST_LENGTH];
        generateChecksum(dataSection, totalSizeOfDataSection, checksum);
        
        chksum = (char *)checksum;
    }


    char senderByte = sender;     // FIXME: api enum here
    char responseByte = response; // FIXME: api send 0x00, svr response with code
    char syncFlagByte = syncFlag;
    int pktIdByte = pktId;

    int numOfParam = m_vDataSection.size();

    size_t pktLen = sizeof(size_t) + (size_t)chkSumSize +
                    sizeof(char) + sizeof(char) +
                    sizeof(char) + sizeof(int) + sizeof(int) +
                    totalSizeOfDataSection;

    char *result = new char[pktLen];
    int totalPktOffset = 0;
    // pkt_len
    memcpy(result, &pktLen, sizeof(size_t));
    totalPktOffset += sizeof(size_t);
    // chksum
    #if TEST_SKT_CHKSUM
        cout << "checksum" << endl;
        printPkt(chksum, SHA256_DIGEST_LENGTH);
    #endif
    if (chksum)
    {
        memcpy(result + totalPktOffset, chksum, SHA256_DIGEST_LENGTH);
        totalPktOffset += SHA256_DIGEST_LENGTH;
    }

    // sender
    memcpy(result + totalPktOffset, &senderByte, sizeof(char));
    totalPktOffset += sizeof(char);
    // response
    memcpy(result + totalPktOffset, &responseByte, sizeof(char));
    totalPktOffset += sizeof(char);
    // sync_flag
    memcpy(result + totalPktOffset, &syncFlagByte, sizeof(char));
    totalPktOffset += sizeof(char);
    // pkt_id
    memcpy(result + totalPktOffset, &pktIdByte, sizeof(int));
    totalPktOffset += sizeof(int);
    // number of parameters
    memcpy(result + totalPktOffset, &numOfParam, sizeof(int));
    totalPktOffset += sizeof(int);
    // data section
    memcpy(result + totalPktOffset, dataSection, totalSizeOfDataSection);
    totalPktOffset += totalSizeOfDataSection;

    // clear data
    clearDataSection();
    if (dataSection)
        delete[] dataSection;

    outLen = totalPktOffset;

    return result;
}

bool ModelSktMsg::verifyChksum(u_char *clnt, u_char *svr)
{
    bool result = true;
    for (size_t i = 0; i < SHA256_DIGEST_LENGTH; i++)
    {
        if (clnt[i] != svr[i])
        {
            result = false;
            break;
        }
    }
    return result;
}

template <typename T>
char *ModelSktMsg::serialize(T data, size_t &outLen)
{
    DType dtype;
    if (typeid(T) == typeid(int))
    {
        dtype = DTYPE_INT;
    }
    else if (typeid(T) == typeid(float))
    {
        dtype = DTYPE_FLOAT;
    }
    else if (typeid(T) == typeid(double))
    {
        dtype = DTYPE_DOUBLE;
    }
    else if (typeid(T) == typeid(char))
    {
        dtype = DTYPE_CHAR;
    }
    else if (typeid(T) == typeid(size_t))
    {
        dtype = DTYPE_SIZE_T;
    }
    else
    {
        printf("[serialize] unsupported type\n");
        return NULL;
    }
    int intDpktSize = sizeof(char) + sizeof(size_t) + sizeof(T);

    char *pkt = new char[intDpktSize];
    memset(pkt, 0, intDpktSize);

    int offset = 0;
    // dType (1bytes)
    char dataType = (char)dtype;
    memcpy(pkt, &dataType, sizeof(dataType));
    offset += sizeof(dataType);

    // data length in bytes (4bytes in integer/ 8 bytes)
    size_t sizeOfData = sizeof(data); // equal to 4
    memcpy(pkt + offset, &sizeOfData, sizeof(size_t));
    offset += sizeof(size_t);

    // store integer data
    memcpy(pkt + offset, &data, sizeof(T));
    offset += sizeof(T);

    outLen = offset;

    m_vDataSection.push_back(
        std::make_pair(pkt, outLen));

    return pkt;
}

template char *ModelSktMsg::serialize<int>(int data, size_t &outLen);
template char *ModelSktMsg::serialize<float>(float data, size_t &outLen);
template char *ModelSktMsg::serialize<double>(double data, size_t &outLen);
template char *ModelSktMsg::serialize<char>(char data, size_t &outLen);
template char *ModelSktMsg::serialize<size_t>(size_t data, size_t &outLen);

template <typename T>
char *ModelSktMsg::serializeArr(T *data, size_t dLen, size_t &outLen)
{
    DType dtype;
    if (typeid(T) == typeid(char))
    {
        dtype = DTYPE_CHAR_ARR;
    }
    else if (typeid(T) == typeid(int))
    {
        dtype = DTYPE_INT_ARR;
    }
    else if (typeid(T) == typeid(float))
    {
        dtype = DTYPE_FLOAT_ARR;
    }
    else if (typeid(T) == typeid(double))
    {
        dtype = DTYPE_DOUBLE_ARR;
    }
    else
    {
        printf("[serializeArr] unsupported type");
        return NULL;
    }

    int farrDpktSize = sizeof(char) + sizeof(size_t) + dLen * sizeof(T);
    char *farrPkt = new char[farrDpktSize];
    memset(farrPkt, 0, farrDpktSize);

    int offset = 0;
    // dType (1bytes)
    char dataType = (char)dtype;
    memcpy(farrPkt, &dataType, sizeof(dataType));
    offset += sizeof(dataType);

    // data length in bytes (N bytes in integer/ 8 bytes)
    size_t sizeOfData = dLen * sizeof(T);
    memcpy(farrPkt + offset, &sizeOfData, sizeof(size_t));
    offset += sizeof(size_t);

    // store float array data
    memcpy(farrPkt + offset, data, dLen * sizeof(T));
    offset += dLen * sizeof(T);

    outLen = offset;
    m_vDataSection.push_back(
        std::make_pair(farrPkt, outLen));

    return farrPkt;
}

template char *ModelSktMsg::serializeArr<char>(char *data, size_t dLen, size_t &outLen);
template char *ModelSktMsg::serializeArr<int>(int *data, size_t dLen, size_t &outLen);
template char *ModelSktMsg::serializeArr<float>(float *data, size_t dLen, size_t &outLen);
template char *ModelSktMsg::serializeArr<double>(double *data, size_t dLen, size_t &outLen);

template <typename T>
T ModelSktMsg::deserialize(const char *data)
{
    T res = 0;
    memcpy(&res, data, sizeof(T));
    return res;
}

template char ModelSktMsg::deserialize<char>(const char *data);
template int ModelSktMsg::deserialize<int>(const char *data);
template float ModelSktMsg::deserialize<float>(const char *data);
template double ModelSktMsg::deserialize<double>(const char *data);
template size_t ModelSktMsg::deserialize<size_t>(const char *data);

template <typename T>
void ModelSktMsg::deserializeArr(T *out, char *pkt, size_t numOfBytes)
{
    memcpy(out, pkt, numOfBytes);
}

template void ModelSktMsg::deserializeArr<char>(char *out, char *pkt, size_t numOfBytes);
template void ModelSktMsg::deserializeArr<u_char>(u_char *out, char *pkt, size_t numOfBytes);
template void ModelSktMsg::deserializeArr<int>(int *out, char *pkt, size_t numOfBytes);
template void ModelSktMsg::deserializeArr<float>(float *out, char *pkt, size_t numOfBytes);
template void ModelSktMsg::deserializeArr<double>(double *out, char *pkt, size_t numOfBytes);

template <typename T>
char *ModelSktMsg::createBatchPkt(T *data, size_t dLen, size_t &outLen)
{
    DType dtype;
    if (typeid(T) == typeid(char))
    {
        dtype = DTYPE_CHAR_ARR;
    }
    else if (typeid(T) == typeid(int))
    {
        dtype = DTYPE_INT_ARR;
    }
    else if (typeid(T) == typeid(float))
    {
        dtype = DTYPE_FLOAT_ARR;
    }
    else if (typeid(T) == typeid(double))
    {
        dtype = DTYPE_DOUBLE_ARR;
    }
    else
    {
        printf("[serializeArr] unsupported type");
        return NULL;
    }

    int pktSize = sizeof(size_t) + dLen;
    outLen = pktSize;
    char *pkt = new char[pktSize];

    memset(pkt, 0x00, pktSize);

    int offset = 0;
    memcpy(pkt, &dLen, sizeof(size_t));
    offset += sizeof(size_t);
    memcpy(pkt + offset, data, sizeof(T) * dLen);
    offset += sizeof(T) * dLen;

    outLen = offset;
    return pkt;
}

template char *ModelSktMsg::createBatchPkt<char>(char *data, size_t dLen, size_t &outLen);
template char *ModelSktMsg::createBatchPkt<int>(int *data, size_t dLen, size_t &outLen);
template char *ModelSktMsg::createBatchPkt<float>(float *data, size_t dLen, size_t &outLen);
template char *ModelSktMsg::createBatchPkt<double>(double *data, size_t dLen, size_t &outLen);
