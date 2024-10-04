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


int main(int argc, char* argv[])
{
    ModelSktMsg msg;
    size_t pktLen = 0;

    int encapsuleData = 99;
    
    char* intpkt = msg.serializeInt(DTYPE_INT, encapsuleData, pktLen);

    cout << "--- Verify Int pkt start ----" << endl;
    cout << "pktLen: " << pktLen << endl;

    for (size_t i = 0; i < pktLen; i++)
    {
        printf("%02x ", intpkt[i]);
        
    }
    printf("\n");

    size_t numOfBytes = extract_size_t(intpkt+1);
    int data = extract_int(intpkt+1+8);

    cout << extract_size_t(intpkt+1) << endl;
    cout << "Verify int: " << (int)(data==encapsuleData) << endl;
    cout << "--- Verify Int pkt End ----" << endl;


    float ftest = 9.9;
    char* floatpkt = msg.serializeFloat(DTYPE_FLOAT, ftest, pktLen);

    cout << "--- Verify float pkt start ----" << endl;
    cout << "pktLen: " << pktLen << endl;

    for (size_t i = 0; i < pktLen; i++)
    {
        printf("%02x ", floatpkt[i]);
        
    }
    printf("\n");
    numOfBytes = extract_size_t(floatpkt+1);
    float fdata = extract_float(floatpkt+1+8);

    cout << extract_size_t(floatpkt+1) << endl;
    cout << "Verify float: " << (int)(fdata==ftest) << endl;
    cout << "--- Verify Float pkt End ----" << endl;

    float fatest[5] = {1.1, 2.2, 3.3, 4.4, 5.5};
    char* farrPkt = msg.serializeFloatArr(DTYPE_FLOAT_ARR, fatest, 5, pktLen);

    cout << "--- Verify float array pkt start ----" << endl;
    
    cout << "pktLen: " << pktLen << endl;

    for (size_t i = 0; i < pktLen; i++)
    {
        printf("%02x ", farrPkt[i]);
    }
    printf("\n");
    numOfBytes = extract_size_t(farrPkt+1);
    cout << "numOfBytes: " << numOfBytes << endl;
    float test[5];
    memcpy(&test, farrPkt+9, 20);
    bool pass = true;
    for (int i = 0; i < 5; i++) {
        if (test[i] != fatest[i]) {
            pass = false;
            break;
        }
    }
    cout << "Verify float array result : " << (int)pass << endl;

    std::vector<std::pair<char*,size_t>>* ds = msg.GetDataSections();
    for (int i = 0; i < ds->size(); i++) {
        cout << (*ds)[i].second << endl;
    }

    msg.createPkt(pktLen);

    cout << "total pkt : " << pktLen << endl;
    return 0;
}