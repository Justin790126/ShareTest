#include <iostream>
#include "ModelSktMsg.h"


using namespace std;

size_t extract_size_t(const char* data) {
    size_t result = 0;
    for (int i = 0; i < sizeof(size_t); ++i) {
        result |= static_cast<unsigned char>(data[i]>>8*i);
    }
    return result;
}

int extract_int(const char* data) {
    int result = 0;
    for (int i = 0; i < sizeof(int); ++i) {
        result |= static_cast<unsigned char>(data[i] >> (8 * i));
    }
    return result;
}

float extract_float(const char* data) {
    float value;
    memcpy(&value, data, sizeof(float));
    return value;
}

double extract_double(const char* data) {
    double val;
    memcpy(&val, data, sizeof(double));
    return val;
}


int main(int argc, char* argv[])
{
    ModelSktMsg msg;
    size_t pktLen = 0;

    int encapsuleData = 99;
    
    char* intpkt = msg.serialize<int>(DTYPE_INT, encapsuleData, pktLen);

    cout << "--- Verify Int pkt start ----" << endl;
    cout << "pktLen: " << pktLen << endl;
    msg.printPkt(intpkt, pktLen);
    printf("\n");

    size_t numOfBytes = extract_size_t(intpkt+1);
    int data = extract_int(intpkt+1+8);

    cout << extract_size_t(intpkt+1) << endl;
    cout << "Verify int: " << (int)(data==encapsuleData) << endl;
    cout << "--- Verify Int pkt End ----" << endl;


    float ftest = 9.9;
    char* floatpkt = msg.serialize<float>(DTYPE_FLOAT, ftest, pktLen);

    cout << "--- Verify float pkt start ----" << endl;
    cout << "pktLen: " << pktLen << endl;
    msg.printPkt(floatpkt, pktLen);
    printf("\n");
    numOfBytes = extract_size_t(floatpkt+1);
    float fdata = extract_float(floatpkt+1+8);

    cout << extract_size_t(floatpkt+1) << endl;
    cout << "Verify float: " << (int)(fdata==ftest) << endl;
    cout << "--- Verify Float pkt End ----" << endl;

    double val = 3.141592543211111;
    char* doublepkt = msg.serialize<double>(DTYPE_DOUBLE, val, pktLen);
    cout << "--- Verify double pkt start ----" << endl;
    cout << "pktLen: " << pktLen << endl;
    msg.printPkt(doublepkt, pktLen);
    numOfBytes = extract_size_t(doublepkt+1);
    double ddata = extract_double(doublepkt+9);
    cout << extract_size_t(doublepkt+1) << endl;
    cout << "Verify double: " << (int)(ddata==val) << endl;
    cout << "--- Verify Double pkt End ----" << endl;



    float fatest[5] = {1.1, 2.2, 3.3, 4.4, 5.5};
    char* farrPkt = msg.serializeArr<float>(DTYPE_FLOAT_ARR, fatest, 5, pktLen);

    cout << "--- Verify float array pkt start ----" << endl;
    
    cout << "pktLen: " << pktLen << endl;
    msg.printPkt(farrPkt, pktLen);
    printf("\n");
    numOfBytes = extract_size_t(farrPkt+1);
    cout << "numOfBytes: " << numOfBytes << endl;
    float test[5];
    memcpy(&test, farrPkt+9, numOfBytes);
    bool pass = true;
    for (int i = 0; i < 5; i++) {
        if (test[i] != fatest[i]) {
            pass = false;
            break;
        }
    }
    cout << "Verify float array result : " << (int)pass << endl;

#ifdef DATA_SECTION_TEST
    std::vector<std::pair<char*,size_t>>* ds = msg.GetDataSections();
    for (int i = 0; i < ds->size(); i++) {
        cout << (*ds)[i].second << endl;
    }
#endif
    cout << "--- Verify double array pkt start ----" << endl;
    double dbtest[5] = {1.1111111111, 2.222222222, 3.33333333, 4.333334, 5.2345678765435};
    char* dbPkt = msg.serializeArr<double>(DTYPE_FLOAT_ARR, dbtest, 5, pktLen);
    msg.printPkt(dbPkt, pktLen);
    numOfBytes = extract_size_t(dbPkt+1);
    cout << "numOfBytes: " << numOfBytes << endl;
    double dbparsed[5];
    memcpy(&dbparsed, dbPkt+9, numOfBytes);
    pass = true;
    for (int i = 0; i < 5; i++) {
        if (dbparsed[i] != dbtest[i]) {
            pass = false;
            break;
        }
    }
    cout << "Verify double array result : " << (int)pass << endl;

    char* pkt = msg.createPkt(pktLen);

    cout << "total pkt : " << pktLen << endl;
    msg.printPkt(pkt, pktLen);
    return 0;
}