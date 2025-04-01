

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


  

  file.close();
  std::cout << "\nEnd of file reached.\n";

  return 0;
}