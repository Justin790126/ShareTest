#include <iostream>
#include "ModelSktMsg.h"

using namespace std;

void ParsePkt(char *pkt)
{
    /*
        1. read pkt_len first
        2. get data section
        3. verify checksum
        4. get response byte
        5. get number of parameters byte
        6. parse data section by for loop
    */
    ModelSktMsg msg;
    size_t pktLen = msg.deserialize<size_t>(pkt);
    cout << "pktLen: " << pktLen << endl;

    int dataOffset = msg.getDataSectionOffset();
    cout << "dataOffset: " << dataOffset << endl;
    int lenDs = pktLen - dataOffset;
    char *ds = new char[lenDs];
    msg.deserializeArr<char>(ds, pkt + dataOffset, lenDs);
    cout << "parsed data section" << endl;
    msg.printPkt(ds, lenDs);
    cout << "----- verify chksum -----" << endl;
    u_char svrChksum[32];
    msg.generateChecksum(ds, lenDs, svrChksum);
    msg.printPkt((char *)svrChksum, 32);
    u_char clntChksum[32];
    msg.deserializeArr<u_char>(clntChksum, pkt + sizeof(size_t), 32);
    bool chksumSame = msg.verifyChksum(clntChksum, svrChksum);
    cout << "chksumSame result : " << chksumSame << endl;

    cout << "--- Response code" << endl;
    char sender = msg.deserialize<char>(pkt + sizeof(size_t) + 32);
    printf("clnt sender: 0x%02x \n", sender);
    char response = msg.deserialize<char>(pkt + sizeof(size_t) + 33);
    printf("svr response: 0x%02x \n", response);

    cout << "--- Number of parameters " << endl;
    int numOfParams = msg.deserialize<int>(pkt + sizeof(size_t) + 34);
    printf(" %d \n", numOfParams);

    cout << "---- deserialize pkt by pkt " << endl;

    // ds
    // lenDs

    int numPara = 0;
    vector<int> endIdxes;
    endIdxes.reserve(numOfParams);
    for (int i = 0; i < lenDs; i++)
    {
        if ((u_char)ds[i] == (u_char)'0xAB') {
            endIdxes.emplace_back(i);
            numPara++;
        }
    }
    printf("num of delimiator %d \n", numPara); // FIXME: choose proper delimtor in pkt
    int st = 0, end=0, numOfdata=0;
    for (size_t i = 0; i < numOfParams; i++)
    {
        
        if (i == 0) {
            st = 0;
            end = endIdxes[i];
            numOfdata = end-st+1;
        } else {
            st = endIdxes[i-1]+1;
            end = endIdxes[i];
            numOfdata = endIdxes[i]-endIdxes[i-1];
        }
        printf("number of bytes: %d :", numOfdata);
        char* data = new char[numOfdata];
        memcpy(data, ds+st, numOfdata);
        msg.printPkt(data, numOfdata);
    }
    
    int pktOffset = 0;
    size_t dataLen = 0;
    double dData = 0.0;
    int iData = 0;
    float fData = 0.0;
    float *fArr = NULL;
    double *dArr = NULL;
    int *iArr = NULL;
    int arrSize = 0;
    while (pktOffset < lenDs)
    {
        u_char ch = ds[pktOffset];
        // cout << pktOffset << "/ " << lenDs << endl;
        switch (ch)
        {
        case DTYPE_INT:
            printf("\n---- parse DTYPE_INT\n");

            dataLen = msg.deserialize<size_t>(ds + pktOffset + 1);
            iData = msg.deserialize<int>(ds + pktOffset + 1 + sizeof(size_t));

            pktOffset += (1 + sizeof(size_t) + dataLen + 1);

            printf("data len: %zu, data: %d\n", dataLen, iData);
            /* code */
            break;
        case DTYPE_FLOAT:
            printf("\n---- parse DTYPE_FLOAT\n");
            // pktOffset++;
            dataLen = msg.deserialize<size_t>(ds + pktOffset + 1);
            fData = msg.deserialize<float>(ds + pktOffset + 1 + sizeof(size_t));

            pktOffset += (1 + sizeof(size_t) + dataLen + 1);
            printf("data len: %zu, data: %f\n", dataLen, fData);
            /* code */
            break;
        case DTYPE_DOUBLE:
            printf("\n---- parse DTYPE_DOUBLE\n");
            /* code */
            dataLen = msg.deserialize<size_t>(ds + pktOffset + 1);
            dData = msg.deserialize<double>(ds + pktOffset + 1 + sizeof(size_t));

            pktOffset += (1 + sizeof(size_t) + dataLen + 1);
            printf("data len: %zu, data: %f\n", dataLen, dData);

            break;
        case DTYPE_INT_ARR:
            /* code */
            printf("\n---- parse DTYPE_INT_ARR\n");
            dataLen = msg.deserialize<size_t>(ds + pktOffset + 1);
            printf("iarr size in bytes: %zu\n", dataLen);
            arrSize = dataLen / sizeof(int);
            printf("iarr size : %d\n", arrSize);
            iArr = new int[arrSize];
            msg.deserializeArr<int>(iArr, ds + pktOffset + 1 + sizeof(size_t), dataLen);
            for (int i = 0; i < arrSize; i++)
                printf("%d ", iArr[i]);
            printf("\n");
            pktOffset += (1 + sizeof(size_t) + dataLen + 1);

            break;
        case DTYPE_FLOAT_ARR:
            /* code */
            printf("\n---- parse DTYPE_FLOAT_ARR\n");
            dataLen = msg.deserialize<size_t>(ds + pktOffset + 1);
            printf("farr size in bytes: %zu\n", dataLen);
            arrSize = dataLen / sizeof(float);
            printf("farr size : %d \n", arrSize);
            fArr = new float[arrSize];
            msg.deserializeArr<float>(fArr, ds + pktOffset + 1 + sizeof(size_t), dataLen);
            for (int i = 0; i < arrSize; i++)
                printf("%f ", fArr[i]);
            printf("\n");

            pktOffset += (1 + sizeof(size_t) + dataLen + 1);
            break;
        case DTYPE_DOUBLE_ARR:
            /* code */
            printf("\n---- parse DTYPE_DOUBLE_ARR\n");
            dataLen = msg.deserialize<size_t>(ds + pktOffset + 1);
            printf("darr size in bytes: %zu\n", dataLen);
            arrSize = dataLen / sizeof(double);
            printf("darr size : %d\n", arrSize);
            dArr = new double[arrSize];
            msg.deserializeArr<double>(dArr, ds + pktOffset + 1 + sizeof(size_t), dataLen);
            for (int i = 0; i < arrSize; i++)
                printf("%.15g ", dArr[i]);
            printf("\n");

            pktOffset += (1 + sizeof(size_t) + dataLen + 1);
            break;

        default:
            printf("--- parse type failed\n");
            pktOffset = lenDs;
            break;
        }
    }

    cout << "---- EOF parsing ----" << endl;
}

int main(int argc, char *argv[])
{
    ModelSktMsg msg;
    size_t pktLen = 0;

    int encapsuleData = 99;

    char *intpkt = msg.serialize<int>(DTYPE_INT, encapsuleData, pktLen);

    cout << "--- Verify Int pkt start ----" << endl;
    cout << "pktLen: " << pktLen << endl;
    msg.printPkt(intpkt, pktLen);
    printf("\n");

    size_t numOfBytes = msg.deserialize<size_t>(intpkt + 1);
    int data = msg.deserialize<int>(intpkt + 1 + 8);

    cout << msg.deserialize<size_t>(intpkt + 1) << endl;
    cout << "Verify int: " << (int)(data == encapsuleData) << endl;
    cout << "--- Verify Int pkt End ----" << endl;

    float ftest = 9.9;
    char *floatpkt = msg.serialize<float>(DTYPE_FLOAT, ftest, pktLen);

    cout << "--- Verify float pkt start ----" << endl;
    cout << "pktLen: " << pktLen << endl;
    msg.printPkt(floatpkt, pktLen);
    printf("\n");
    numOfBytes = msg.deserialize<size_t>(floatpkt + 1);
    float fdata = msg.deserialize<float>(floatpkt + 1 + 8);

    cout << msg.deserialize<size_t>(floatpkt + 1) << endl;
    cout << "Verify float: " << (int)(fdata == ftest) << endl;
    cout << "--- Verify Float pkt End ----" << endl;

    double val = 3.141592543211111;
    char *doublepkt = msg.serialize<double>(DTYPE_DOUBLE, val, pktLen);
    cout << "--- Verify double pkt start ----" << endl;
    cout << "pktLen: " << pktLen << endl;
    msg.printPkt(doublepkt, pktLen);
    numOfBytes = msg.deserialize<size_t>(doublepkt + 1);
    double ddata = msg.deserialize<double>(doublepkt + 9);
    cout << msg.deserialize<size_t>(doublepkt + 1) << endl;
    cout << "Verify double: " << (int)(ddata == val) << endl;
    cout << "--- Verify Double pkt End ----" << endl;

    float fatest[5] = {1.1, 2.2, 3.3, 4.4, 5.5};
    char *farrPkt = msg.serializeArr<float>(DTYPE_FLOAT_ARR, fatest, 5, pktLen);

    cout << "--- Verify float array pkt start ----" << endl;

    cout << "pktLen: " << pktLen << endl;
    msg.printPkt(farrPkt, pktLen);
    printf("\n");
    numOfBytes = msg.deserialize<size_t>(farrPkt + 1);
    cout << "numOfBytes: " << numOfBytes << endl;
    float test[5];
    msg.deserializeArr<float>(test, farrPkt + 9, numOfBytes);
    bool pass = true;
    for (int i = 0; i < 5; i++)
    {
        if (test[i] != fatest[i])
        {
            pass = false;
            break;
        }
    }
    cout << "Verify float array result : " << (int)pass << endl;

#ifdef DATA_SECTION_TEST
    std::vector<std::pair<char *, size_t>> *ds = msg.GetDataSections();
    for (int i = 0; i < ds->size(); i++)
    {
        cout << (*ds)[i].second << endl;
    }
#endif
    cout << "--- Verify double array pkt start ----" << endl;
    double dbtest[5] = {1.1111111111, 2.222222222, 3.33333333, 4.333334, 5.2345678765435};
    char *dbPkt = msg.serializeArr<double>(DTYPE_DOUBLE_ARR, dbtest, 5, pktLen);
    msg.printPkt(dbPkt, pktLen);
    cout << "pktLen: " << pktLen << endl;
    numOfBytes = msg.deserialize<size_t>(dbPkt + 1);
    cout << "numOfBytes: " << numOfBytes << endl;
    double dbparsed[5];
    msg.deserializeArr<double>(dbparsed, dbPkt + 9, numOfBytes);
    pass = true;
    for (int i = 0; i < 5; i++)
    {
        if (dbparsed[i] != dbtest[i])
        {
            pass = false;
            break;
        }
    }
    cout << "Verify double array result : " << (int)pass << endl;

    cout << "--- Verify int array pkt start ----" << endl;
    int ittest[5] = {1, 3, 5, 7, 9};
    char *ipkt = msg.serializeArr<int>(DTYPE_INT_ARR, ittest, 5, pktLen);
    msg.printPkt(ipkt, pktLen);
    numOfBytes = msg.deserialize<size_t>(ipkt + 1);
    cout << "pktLen: " << pktLen << endl;
    cout << "numOfBytes: " << numOfBytes << endl;
    int iparsed[5];
    msg.deserializeArr<int>(iparsed, ipkt + 9, numOfBytes);
    pass = true;
    for (int i = 0; i < 5; i++)
    {
        if (iparsed[i] != ittest[i])
        {
            pass = false;
            break;
        }
    }
    cout << "Verify int array result : " << (int)pass << endl;

    cout << "Verify total packet: " << endl;
    char *pkt = msg.createPkt(pktLen);

    cout << "total pkt : " << pktLen << endl;
    msg.printPkt(pkt, pktLen);

    cout << "--------- Start parse all packet ---------" << endl;

    ParsePkt(pkt);
    return 0;
}