#include <iostream>
#include <string>
#include <cstdlib>
#include <unistd.h>  // getopt()
#include <stdio.h>
#include <getopt.h>

using namespace std;
 
int main(int argc, char** argv)
{
    // use getopt to get -i option -m option -o option
    int opt;
    std::string input_file, output_file;
    int mode=-1;

    while ((opt = getopt(argc, argv, "i:m:o:"))!= -1) {
        switch(opt) {
            case 'i':
                input_file = optarg;
                break;
            case 'm':
                mode = atoi(optarg);
                break;
            case 'o':
                output_file = optarg;
                break;
            default:
                std::cerr << "Usage: " << argv[0] << " -i <input_file> -m <mode> -o <output_file>" << std::endl;
                return 1;
        }
    }

    if (input_file.empty() || mode==-1 || output_file.empty()) {
        std::cerr << "All options (-i, -m, -o) must be provided." << std::endl;
        return 1;
    }

    printf("Input file: %s\n", input_file.c_str());
    printf("Mode: %d\n", mode);
    printf("Output file: %s\n", output_file.c_str());
}