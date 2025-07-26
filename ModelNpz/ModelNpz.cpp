#include "ModelNpz.h"

void ModelNpz::run()
{
    m_fp = new ifstream(m_sFname, ios::binary);
    if (!m_fp->is_open()) {
        cerr << "Error opening file: " << m_sFname << endl;
        return;
    }

    bool is_npz = false;
    char byte;
    while (m_fp->get(byte)) {
        // if byte == 0x93 break
        uchar ub = static_cast<unsigned char>(byte);
        if (ub == m_ucMagicByte) {
            char buffer[5];
            m_fp->read(buffer, 5);
            if (strcmp(buffer, m_sMagic.c_str())==0) {
                is_npz = true;
                break;
            }
        }
    }

    if (!is_npz) {
        cerr << "File is not in npz format." << endl;
        return;
    } else {
        cout << "File is in npz format." << endl;
    }

    m_sNpVer = "";
    if (m_fp->get(byte)) {
        if ((uchar)byte == 0x01) {
            m_sNpVer = "1";
        } else {
            cerr << "Unsupported npz format version." << endl;
            return;
        }   
    }

    if (m_fp->get(byte)) {
        if ((uchar)byte == 0x00) {
            m_sNpVer += ".0";
        } else {
            cerr << "Unsupported npz format version." << endl;
            return;
        }
    }

    cout << "Numpy version : " << m_sNpVer << endl;

    /*
        Parse header length
    */
    unsigned short int headerLen = 0;
    char headerLenBytes[2];
    if (m_fp->read(headerLenBytes, 2)) {
        memcpy(&headerLen, headerLenBytes, sizeof(headerLen));
        cout << "Header length: " << headerLen << endl;
    }

    if (headerLen == 0) {
        cerr << "Header length is zero, no data to read." << endl;
        return;
    }

    /*
        Read all header bytes
    */
    char* header = new char[headerLen];
    if (m_fp->read(header, headerLen)) {
        // print header in hex
        
    } else {
        cerr << "Error reading header." << endl;
        delete[] header;
        return;
    }

    /*
        Parse header to string
    */
    string paramTuple;
    for (int i = 0; i < headerLen; ++i) {
        // printf("%02x ", static_cast<unsigned char>(header[i]));
        if ((uchar)header[i] == 0x0A) {
            break;
        } else {
            if ((uchar)header[i] != 0x20) {
                paramTuple += header[i];
            }
        }
    }
    cout << "\nParameter tuple: " << paramTuple << endl;

    /*
        Parse parameter tuple to map
          Example: {'descr':'|u1','fortran_order':False,'shape':(512,512,3),}
    */

    

    // parse tuple by :
    m_NpzHeader.clear();
    size_t posDesc = paramTuple.find(m_sDescKey);
    size_t posFortran = paramTuple.find(m_sFortranKey);
    if (posDesc != string::npos && posFortran != string::npos) {
        int st = posDesc + m_sDescKey.length() + 3;
        int en = posFortran - 3;
        string descVal = paramTuple.substr(st, en - st);
        m_NpzHeader[m_sDescKey] = descVal;
        cout << "Description: " << descVal << endl;
    } else {
        cerr << "Description or Fortran order key not found in parameter tuple." << endl;
    }

    size_t posShape = paramTuple.find(m_sShapeKey);
    if (posFortran != string::npos && posShape != string::npos) {
        string fortranVal = paramTuple.substr(posFortran + m_sFortranKey.length() + 2, 5);
        cout << "Fortran order: " << fortranVal << endl;
        m_NpzHeader[m_sFortranKey] = fortranVal;
        
    } else {
        cerr << "Shape key not found in parameter tuple." << endl;
    }

    if (posShape != string::npos) {
        int st = posShape + m_sShapeKey.length() + 3; // +3 to skip the key, the colon, and the opening parenthesis
        int en = paramTuple.find(')', st); // find the closing parenthesis
        if (en != string::npos) {
            string shapeVal = paramTuple.substr(st, en - st);
            cout << "Shape: " << shapeVal << endl;
            m_NpzHeader[m_sShapeKey] = shapeVal;
        } else {
            cerr << "Closing parenthesis for shape not found." << endl;
        }
    } else {
        cerr << "Shape key not found in parameter tuple." << endl;
    }



    if (m_NpzHeader.count(m_sDescKey) != 0) {
        if (m_NpzHeader[m_sDescKey].find("u1") != string::npos) {
            cout << "Data type: unsigned 8-bit integer (u1)" << endl;
        } else if (m_NpzHeader[m_sDescKey].find("f4") != string::npos) {
            cout << "Data type: 32-bit float (f4)" << endl;
        } else if (m_NpzHeader[m_sDescKey].find("f8") != string::npos) {
            cout << "Data type: 64-bit float (f8)" << endl;
        } else {
            cout << "Data type: " << m_NpzHeader[m_sDescKey] << endl;
        }
    } else {
        cerr << "No description found in header." << endl;
    }
}