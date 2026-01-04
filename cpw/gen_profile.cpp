#include <fstream>
#include <string>
#include <iostream>

void write_cell(std::ofstream& ofs, std::string name, double l, double b, double r, double t) {
    unsigned int ref = 0, nlen = name.length();
    ofs.write((char*)&ref, 4); ofs.write((char*)&nlen, 4);
    ofs.write(name.c_str(), nlen);
    ofs.write((char*)&l, 8); ofs.write((char*)&b, 8);
    ofs.write((char*)&r, 8); ofs.write((char*)&t, 8);
    unsigned long long d64 = 0; unsigned char d8 = 1;
    ofs.write((char*)&d64, 8); ofs.write((char*)&d64, 8);
    ofs.write((char*)&d64, 8); ofs.write((char*)&d8, 1);
}

int main() {
    std::ofstream ofs("test.cellprofile", std::ios::binary);
    std::string ver = "Binary_5.2";
    unsigned int vlen = ver.length(), pid = 1234;
    ofs.write((char*)&vlen, 4); ofs.write(ver.c_str(), vlen);
    ofs.write((char*)&pid, 4);

    // 生成三個 Top Cell
    write_cell(ofs, "TOP_A_CORE", 0, 0, 5000, 5000);
    write_cell(ofs, "TOP_B_IO", 10000, 0, 12000, 5000);
    write_cell(ofs, "TOP_C_SRAM", 0, 10000, 2000, 15000);

    for(int i=0; i<100; ++i) 
        write_cell(ofs, "leaf_"+std::to_string(i), i, i, i+5, i+5);
    
    std::cout << "Generated test.cellprofile with 3 TOP cells." << std::endl;
    return 0;
}