#include <fstream>
#include <string>

void write_raw(std::ofstream& ofs, std::string name, double l, double b, double r, double t) {
    unsigned int ref = 0, nlen = name.length();
    ofs.write((char*)&ref, 4);
    ofs.write((char*)&nlen, 4);
    ofs.write(name.c_str(), nlen);
    ofs.write((char*)&l, 8); ofs.write((char*)&b, 8);
    ofs.write((char*)&r, 8); ofs.write((char*)&t, 8);
    unsigned long long p=0, fi=0, fc=0; unsigned char fl=1;
    ofs.write((char*)&p, 8); ofs.write((char*)&fi, 8);
    ofs.write((char*)&fc, 8); ofs.write((char*)&fl, 1);
}

int main(int argc, char** argv) {
    std::ofstream ofs("test.cellprofile", std::ios::binary);
    std::string ver = "Binary_5.2";
    unsigned int vlen = ver.length(), pid = 123;
    ofs.write((char*)&vlen, 4); ofs.write(ver.c_str(), vlen);
    ofs.write((char*)&pid, 4);

    // 第一筆作為 TopCell
    write_raw(ofs, "TOP_CELL", 0, 0, 5000, 5000);
    // 產生 5000+ 筆子 Cell
    for(int i=0; i<5500; ++i) {
        write_raw(ofs, "Sub_" + std::to_string(i), i, i, i+10, i+10);
    }
    return 0;
}