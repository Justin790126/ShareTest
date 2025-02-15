#include "ModelSktBase.h"


void ModelSktBase::Send(char* pkt, size_t pktLen)
{
    int sendbytes = send(client_socket, pkt, pktLen, 0);
    if (sendbytes < 0) {
        char resMsg[128];
        sprintf(resMsg, "[ModelSktBase] Send failed with byte len %d", sendbytes);
        m_sStatusMsg = std::move(resMsg);
        return;
    }
    // printf("[ModelSktBase] send %d bytes\n", sendbytes);
}

size_t ModelSktBase::BatchReceive(float* img)
{
    size_t recvBytes = 0;
    char resMsg[1024];
    ModelSktMsg msg;
    size_t gid = 0;
   
    char sizeBytes[8];
    int bytes_received = recv(client_socket, sizeBytes, sizeof(size_t), 0);
    if (bytes_received <= 0)
    {
        sprintf(resMsg, "[ModelSktBase] Batch Receive size bytes failed");
        m_sStatusMsg = std::move(resMsg);
        return -1;
    }
    size_t pktLen = msg.deserialize<size_t>(sizeBytes);

    char* arr = new char[pktLen];
    bytes_received = recv(client_socket, arr, pktLen, 0);
    if (bytes_received <= 0) {
        sprintf(resMsg, "[ModelSktBase] Batch Receive size bytes failed");
        m_sStatusMsg = std::move(resMsg);
        return -1;
    }
    printf("gid %zu: , recv pkt = %zu\n", gid, pktLen);
    memcpy(img+recvBytes, arr, pktLen);
    if (arr) delete[] arr;
    recvBytes += pktLen;
    gid++;
    return recvBytes;
}

bool ModelSktBase::Receive(vector<PktRes>& oRes)
{
    bool result = false;
    char resMsg[1024];
    ModelSktMsg msg;
    int verbose = m_iVerbose;

    char sizeBytes[8];
    int bytes_received = recv(client_socket, sizeBytes, sizeof(size_t), 0);
    if (bytes_received <= 0)
    {
        sprintf(resMsg, "[ModelSktBase] Receive size bytes failed");
        m_sStatusMsg = std::move(resMsg);
        return result; // Connection closed or error
    }
    size_t pktLen = msg.deserialize<size_t>(sizeBytes);

    if (verbose > 0) {
        printf("[ModelSktBase] pktLen = %zu \n", pktLen);
    }

    size_t remainPkt = pktLen - sizeof(size_t);
    if (m_pPkt) delete []m_pPkt;
    m_pPkt = new char[pktLen];
    memcpy(m_pPkt, sizeBytes, sizeof(size_t));
    bytes_received = recv(client_socket, m_pPkt+sizeof(size_t), remainPkt, 0);
    if (bytes_received <= 0)
    {
        sprintf(resMsg, "[ModelSktBase] Receive remain all bytes failed");
        m_sStatusMsg = std::move(resMsg);
        return result; // Connection closed or error
    }
    // if (verbose > 0) {
    //     // msg.printPkt(m_pPkt, pktLen);
    // }

    // // Get data section
    char* pkt = m_pPkt;
    int dataOffset = msg.getDataSectionOffset();    
    int lenDs = pktLen - dataOffset;
    char *ds = new char[lenDs];
    msg.deserializeArr<char>(ds, pkt + dataOffset, lenDs);
    // // Verify checksum
    u_char svrChksum[32];
    msg.generateChecksum(ds, lenDs, svrChksum);
    if (verbose > 0) {
        cout << "checksum generate from data section " << endl;
        msg.printPkt((char*)svrChksum, 32);
    }
    
    // // if (verbose > 0) msg.printPkt((char *)svrChksum, 32);
    u_char clntChksum[32];
    msg.deserializeArr<u_char>(clntChksum, pkt + sizeof(size_t), 32);
    if (verbose > 0) {
        cout << "clntChksum:" << endl;
        msg.printPkt((char*)clntChksum, 32);
    }
    
    bool chksumSame = msg.verifyChksum(clntChksum, svrChksum);
    // if (verbose > 0)
    // {
    //     cout << "dataOffset: " << dataOffset << endl;
    //     cout << "parsed data section" << endl;
    //     // msg.printPkt(ds, lenDs);
    //     cout << "----- verify chksum -----" << endl;
    //     cout << "chksumSame result : " << chksumSame << endl;
    // }
    if (!chksumSame) {
        sprintf(resMsg, "[ModelSktBase] Verify checksum failed");
        m_sStatusMsg = std::move(resMsg);
        return result;
    }

    // // Verify status code
    int headerOffset = msg.getChksumSize();
    char sender = msg.deserialize<char>(pkt + sizeof(size_t) + headerOffset);
    headerOffset += sizeof(char);
    char response = msg.deserialize<char>(pkt + sizeof(size_t) + headerOffset);
    headerOffset += sizeof(char);
    char syncFlag = msg.deserialize<char>(pkt + sizeof(size_t) + headerOffset);
    headerOffset += sizeof(char);
    int pktid = msg.deserialize<int>(pkt + sizeof(size_t) + headerOffset);
    headerOffset += sizeof(int);
    if (verbose > 1) {
        printf("clnt sender: 0x%02x \n", sender);
        printf("svr response: 0x%02x \n", response);
        printf("sync flag: 0x%02x \n", syncFlag);
        printf("pktid: %x \n", pktid);
    }

    // // Verify number of parameters
    int numOfParams = msg.deserialize<int>(pkt + sizeof(size_t) + headerOffset);

    int pktOffset = 0;
    size_t dataLen = 0;

    float *fArr = NULL;
    double *dArr = NULL;
    int *iArr = NULL;
    char* cArr=NULL;
    int arrSize = 0;
    oRes.reserve(numOfParams);
    while (pktOffset < lenDs)
    {
        u_char ch = ds[pktOffset];
        
        PktRes param;
        param.dType = (DType)ch;
        param.cSender = sender;
        param.cResCode = response;
        param.cSyncFlg = syncFlag;
        param.pktId = pktid;
        if (verbose > 0) printf("\n===> offset/ lenDs, type = %d/ %d, 0x%02x ", pktOffset, lenDs, param.dType);

        switch (ch)
        {
        case DTYPE_INT:
            dataLen = msg.deserialize<size_t>(ds + pktOffset + 1);
            param.iData = msg.deserialize<int>(ds + pktOffset + 1 + sizeof(size_t));
            pktOffset += (1 + sizeof(size_t) + dataLen + 1);
            oRes.emplace_back(param);

            if (verbose > 1) {
                printf("---- parse DTYPE_INT\n");
                printf("data len: %zu, data: %d\n", dataLen, param.iData);
            }
            break;
        case DTYPE_CHAR:
            dataLen = msg.deserialize<size_t>(ds + pktOffset + 1);
            param.cData = msg.deserialize<char>(ds + pktOffset + 1 + sizeof(size_t));
            pktOffset += (1 + sizeof(size_t) + dataLen + 1);
            oRes.emplace_back(param);
            if (verbose > 1) {
                printf("---- parse DTYPE_CHAR\n");
                printf("data len: %zu, data: 0x%02x\n", dataLen, param.cData);
            }
            break;
        case DTYPE_FLOAT:
            // pktOffset++;
            dataLen = msg.deserialize<size_t>(ds + pktOffset + 1);
            param.fData = msg.deserialize<float>(ds + pktOffset + 1 + sizeof(size_t));
            pktOffset += (1 + sizeof(size_t) + dataLen + 1);
            oRes.emplace_back(param);
            if (verbose > 1) {
                printf("---- parse DTYPE_FLOAT\n");
                printf("data len: %zu, data: %f\n", dataLen, param.fData);
            }
            break;
        case DTYPE_DOUBLE:
            dataLen = msg.deserialize<size_t>(ds + pktOffset + 1);
            param.dData = msg.deserialize<double>(ds + pktOffset + 1 + sizeof(size_t));
            pktOffset += (1 + sizeof(size_t) + dataLen + 1);
            oRes.emplace_back(param);
            if (verbose > 1) {
                printf("---- parse DTYPE_DOUBLE\n");
                printf("data len: %zu, data: %f\n", dataLen, param.dData);
            }
            break;
        case DTYPE_SIZE_T:
            dataLen = msg.deserialize<size_t>(ds + pktOffset + 1);
            param.sData = msg.deserialize<size_t>(ds + pktOffset + 1 + sizeof(size_t));
            pktOffset += (1 + sizeof(size_t) + dataLen + 1);
            oRes.emplace_back(param);
            if (verbose > 1) {
                printf("---- parse DTYPE_SIZE_T\n");
                printf("data len: %zu, data: %f\n", dataLen, param.sData);
            }
            break;
        case DTYPE_INT_ARR:
            dataLen = msg.deserialize<size_t>(ds + pktOffset + 1);
            arrSize = dataLen / sizeof(int);
            iArr = new int[arrSize];
            msg.deserializeArr<int>(iArr, ds + pktOffset + 1 + sizeof(size_t), dataLen);
            param.arrSize = arrSize;
            param.arr = (void*)iArr;
            pktOffset += (1 + sizeof(size_t) + dataLen + 1);
            oRes.emplace_back(param);
            if (verbose > 1) {
                printf("---- parse DTYPE_INT_ARR\n");
                printf("iarr size in bytes: %zu\n", dataLen);
                printf("iarr size : %d\n", arrSize);
                for (int i = 0; i < arrSize; i++)
                    printf("%d ", iArr[i]);
                printf("\n");
            }
            break;
        case DTYPE_FLOAT_ARR:
            dataLen = msg.deserialize<size_t>(ds + pktOffset + 1);
            arrSize = dataLen / sizeof(float);
            fArr = new float[arrSize];
            msg.deserializeArr<float>(fArr, ds + pktOffset + 1 + sizeof(size_t), dataLen);
            param.arr = (void*)fArr;
            param.arrSize = arrSize;
            pktOffset += (1 + sizeof(size_t) + dataLen + 1);
            oRes.emplace_back(param);
            // if (verbose > 0) {
            //     printf("---- parse DTYPE_FLOAT_ARR\n");
            //     printf("farr size in bytes: %zu\n", dataLen);
            //     printf("farr size : %d \n", arrSize);
            //     for (int i = 0; i < arrSize; i++)
            //         printf("%f ", fArr[i]);
            //     printf("\n");
            // }
            break;
        case DTYPE_DOUBLE_ARR:
            dataLen = msg.deserialize<size_t>(ds + pktOffset + 1);
            arrSize = dataLen / sizeof(double);
            dArr = new double[arrSize];
            msg.deserializeArr<double>(dArr, ds + pktOffset + 1 + sizeof(size_t), dataLen);
            param.arr = (void*)dArr;
            param.arrSize = arrSize;
            pktOffset += (1 + sizeof(size_t) + dataLen + 1);
            oRes.emplace_back(param);
            if (verbose > 1) {
                printf("---- parse DTYPE_DOUBLE_ARR\n");
                printf("darr size in bytes: %zu\n", dataLen);
                printf("darr size : %d\n", arrSize);
                for (int i = 0; i < arrSize; i++)
                    printf("%.15g ", dArr[i]);
                printf("\n");
            }
            break;
        case DTYPE_CHAR_ARR:
            dataLen = msg.deserialize<size_t>(ds + pktOffset + 1);
            arrSize = dataLen / sizeof(char);
            cArr = new char[arrSize];
            msg.deserializeArr<char>(cArr, ds + pktOffset + 1 + sizeof(size_t), dataLen);
            param.arrSize = arrSize;
            param.arr = (void*)cArr;
            pktOffset += (1 + sizeof(size_t) + dataLen + 1);
            oRes.emplace_back(param);
            if (verbose > 1) {
                printf("---- parse DTYPE_CHAR_ARR\n");
                printf("darr size in bytes: %zu\n", dataLen);
                printf("darr size : %d\n", arrSize);
                printf("result : %s \n", cArr);
            }
            break;

        default:
            pktOffset = lenDs;
            sprintf(resMsg, "[ModelSktBase] Parse data failed");
            m_sStatusMsg = std::move(resMsg);
            return result;
        }
    }
    sprintf(resMsg, "[ModelSktBase] EOF parsing");
    m_sStatusMsg = std::move(resMsg);

    if (m_pPkt) delete[] m_pPkt;
    m_pPkt = NULL;

    result = true;
    return result;
}
