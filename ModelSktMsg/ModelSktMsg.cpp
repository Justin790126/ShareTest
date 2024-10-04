#include "ModelSktMsg.h"


ModelSktMsg::ModelSktMsg(/* args */)
{
    m_vDataSection.clear();
}

void ModelSktMsg::ClearDataSection()
{
    for (int i = 0; i < m_vDataSection.size(); i++) {
        if (m_vDataSection[i].first) {
            delete [] m_vDataSection[i].first;
        }
    }
    m_vDataSection.clear();
}

ModelSktMsg::~ModelSktMsg()
{
    ClearDataSection();
}

void ModelSktMsg::printPkt(char* pkg, size_t dsize)
{
    for (size_t i = 0; i < dsize; i++)
    {
        printf("%02x ", (u_char)pkg[i]);
    }
    printf("\n");
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

    m_vDataSection.push_back(
        std::make_pair(intPkt, outLen)
    );
    
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
    m_vDataSection.push_back(
        std::make_pair(intPkt, outLen)
    );
    
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
    m_vDataSection.push_back(
        std::make_pair(farrPkt, outLen)
    );
    
    return farrPkt;
}

void ModelSktMsg::generateChecksum(char* data, size_t sizeOfData, u_char* chksum)
{
    SHA256(reinterpret_cast<const unsigned char*>(data), sizeOfData, chksum);
}

char* ModelSktMsg::createPkt(size_t& outLen)
{
    if (m_vDataSection.empty()) return NULL;

    size_t totalSizeOfDataSection = 0;
    for (int i = 0; i < m_vDataSection.size(); i++) {
        totalSizeOfDataSection += m_vDataSection[i].second;
    }

    char* dataSection = new char[totalSizeOfDataSection];
    memset(dataSection, 0, totalSizeOfDataSection);
    int offset = 0;
    for (int i = 0; i < m_vDataSection.size(); i++) {
        char* data = m_vDataSection[i].first;
        memmove(dataSection+offset, data, m_vDataSection[i].second);
        offset+=m_vDataSection[i].second;
    }

    printf("----total data section----\n");
    printPkt(dataSection, offset);
    printf("\n");



    u_char checksum[SHA256_DIGEST_LENGTH];
    generateChecksum(dataSection, totalSizeOfDataSection, checksum);
    // Calculate the SHA-256 checksum
    printf("---checksum----\n");
    char* chksum = (char*)checksum;
    printPkt(chksum, SHA256_DIGEST_LENGTH);
    printf("\n");

    char sender = 0x03;
    char response = 0x05;
    int numOfParam = m_vDataSection.size();

    size_t pktLen = sizeof(size_t) + (size_t)SHA256_DIGEST_LENGTH + sizeof(char) + sizeof(char) + sizeof(int) + totalSizeOfDataSection;


    char* result = new char[pktLen];
    int totalPktOffset=0;
    // pkt_len
    memcpy(result, &pktLen, sizeof(size_t));
    totalPktOffset += sizeof(size_t);
    // chksum
    memcpy(result+totalPktOffset, &checksum, SHA256_DIGEST_LENGTH);
    totalPktOffset += SHA256_DIGEST_LENGTH;
    // sender
    memcpy(result+totalPktOffset, &sender, sizeof(char));
    totalPktOffset += sizeof(char);
    // response
    memcpy(result+totalPktOffset, &response, sizeof(char));
    totalPktOffset += sizeof(char);
    // number of parameters
    memcpy(result+totalPktOffset, &numOfParam, sizeof(int));
    totalPktOffset += sizeof(int);
    // data section
    memcpy(result+totalPktOffset, dataSection, totalSizeOfDataSection);
    totalPktOffset += totalSizeOfDataSection;

    printf("----total size: %d----\n", totalPktOffset);
    printPkt(result, totalPktOffset);
    printf("\n");

    // clear data
    ClearDataSection();
    if (dataSection) delete []dataSection;

    outLen = totalPktOffset;

    
    return result;
}