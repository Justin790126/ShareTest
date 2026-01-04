#include <fstream>
#include <string>
#include <vector>
#include <stdint.h>
#include <unistd.h>

void profile_write_name(std::ofstream& ofs, const std::string& name) {
    uint32_t len = name.length();
    ofs.write((char*)&len, sizeof(uint32_t));
    ofs.write(name.c_str(), len);
}

int main() {
    std::ofstream ofs("test.cellprofile", std::ios::binary);
    if(!ofs) return 1;

    // 1. Header
    profile_write_name(ofs, "Binary_5.2");
    uint32_t pid = getpid(), dbu = 1000;
    uint64_t table_off = 0;
    ofs.write((char*)&pid, 4);
    ofs.write((char*)&dbu, 4);
    ofs.write((char*)&table_off, 8);

    // 2. Cell Records (生成 6000 個)
    std::vector<std::string> topNames = {"TOP_A", "TOP_B", "TOP_C"};
    for(int i=0; i<6000; ++i) {
        unsigned char type = 1;
        ofs.write((char*)&type, 1);
        
        std::string name;
        if(i < 3) name = topNames[i];
        else name = "leaf_" + std::to_string(i);

        uint32_t cindex = i;
        ofs.write((char*)&cindex, 4);
        profile_write_name(ofs, name);
        
        double coord = (double)i;
        ofs.write((char*)&coord, 8); ofs.write((char*)&coord, 8); // L, B
        coord += 10.0;
        ofs.write((char*)&coord, 8); ofs.write((char*)&coord, 8); // R, T
        
        uint64_t zero8 = 0; uint32_t zero4 = 0; unsigned char zero1 = 0;
        ofs.write((char*)&zero8, 8); ofs.write((char*)&zero8, 8); // pos, len
        ofs.write((char*)&zero8, 8); ofs.write((char*)&zero4, 4); // fig_cnt, used
        ofs.write((char*)&zero1, 1); // flag
    }

    // 3. End Tag
    unsigned char end_tag = 0;
    ofs.write((char*)&end_tag, 1);

    // 4. Top Cells (寫入 3 個)
    uint32_t top_n = 3;
    ofs.write((char*)&top_n, 4);
    for(const auto& name : topNames) {
        profile_write_name(ofs, name);
    }

    // 5. Layer Mapping (0層)
    uint32_t lay_n = 0;
    ofs.write((char*)&lay_n, 4);
    
    // 7. Footer
    ofs.write((char*)&pid, 4);
    return 0;
}