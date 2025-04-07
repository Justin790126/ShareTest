

#include <iostream>
#include <fstream>
#include <cstring>
#include <memory>
using namespace std;
// drcSuiteTests_au3.oas

static const char magic_bytes[] = { "%SEMI-OASIS" };
static const char ver_bytes[] = { "1.0" };

const char CR = 0x0D;
const char NL = 0x0A;

string GetStr(ifstream& file)
{
  char ch;
  if (file.get(ch)) {
    size_t strSize = static_cast<size_t>(static_cast<unsigned char>(ch));
    
    std::shared_ptr<char> strbuf(new char[strSize], std::default_delete<char[]>());

    if (file.read(strbuf.get(), strSize)) {
      return string(strbuf.get());
    }
  }
  return "";
}

template<typename T>
T GetUint(std::ifstream& file)
{
    T v = 0;
    T vm = 1;
    char c;
    do {
        file.get(c);
        // printf("0x%02x ", (unsigned char)c);
        v += static_cast<T>(c & 0x7f) * vm;
        vm <<= 7;
    } while ((c & 0x80) != 0);

    return v;
}

double GetReal(ifstream& file)
{
  unsigned int typeB = GetUint<unsigned int>(file);
  unsigned int type = typeB & 0x07;
  // cout << "type: " << type << endl;
  if (type == 0) {
    return double(GetUint<unsigned long>(file));
  }

  return -1;
}

int main(int argc, char **argv) {
  string fname = argv[1];
  cout << fname << endl;
  std::ifstream file(fname, std::ios::binary);
  if (!file) {
    std::cerr << "Error opening file!\n";
    return 1;
  }

  const int sizeOfMgb = sizeof(magic_bytes) - 1;
  std::shared_ptr<char> buffer(new char[sizeOfMgb], std::default_delete<char[]>());

  bool isOasis = false;
  if (file.read(buffer.get(), sizeOfMgb)) {
    if (strcmp(buffer.get(), magic_bytes) == 0) {
      cout << "Get magic byte success" << endl;
      isOasis = true;
    }
  }

  if (!isOasis) return -1;

  char ch;

  if (file.get(ch)) {
    if (ch == CR) {
      cout << "[CR] hit" << endl;
    }
  }

  if (file.get(ch)) {
    if (ch == NL) {
      cout << "[NL] hit" << endl;
    }
  }

  if (file.get(ch)) {
    if (ch == 0x01) {
      cout << "[START] Tag hit" << endl;
    } 
  }

  // realize get_str
  string oasVer = GetStr(file);
  cout << "Oas version: " << oasVer << endl;
  if (oasVer != "1.0") return -1;

  double res = GetReal(file);
  cout << "1dbu = " << res << "um" << endl;

  bool table_offsets_at_end = GetUint<unsigned int>(file);
  if (!table_offsets_at_end) {
    cout << "table offset not at end" << endl;
    exit(-1);
  }
  printf("\n");

  // file.get(ch);

  while(file.get(ch)) {

    // printf("0x%02x ", (unsigned char)ch);

    if (ch == 0 /*PAD*/) {
      cout << "[PAD] hit" << endl;
      // file.get(ch);
    } else if (ch == 0x02 /*END*/) {
      cout << "[END] hit" << endl;
      break;
    } else if (ch ==3 || ch == 4 /*CELLNAME*/) {
      std::string name = GetStr(file);
      cout << "Cell name: " << name << endl;
    }else if (ch== 5 || ch == 6 /*TEXTSTRING*/) {
      std::string name = GetStr(file);
      cout << "Text string: " << name << endl;
    }else if (ch == 7 || ch == 8 /*PROPNAME*/) {
      
      std::string name = GetStr(file);
      cout << "propname: " << name << endl;

    } else if (ch == 9 || ch == 10 /*PROPSTRING*/) {

    } else if (ch == 11 || ch == 12 /*LAYERNAME*/) {

    }  else if (ch == 28 || ch == 29 /*PROPERTY*/) {

    } else if (ch == 30 || ch == 31 /*XNAME*/) {

    } else if (ch == 13 || ch == 14 /*CELL*/) {

      cout << "Cell" << endl;

      break;

    } else if (ch == 34 /*CBLOCK*/) {
    
    } else {

    }

  }
  

  file.close();
  std::cout << "\nEnd of file reached.\n";

  return 0;
}