#include <QtGui>
#include <iostream>
#include <string>
#include <cstdlib>
#include <unistd.h> // getopt()
#include <stdio.h>
#include <getopt.h>

#include "ModelCrypto.h"

using namespace std;

int main(int argc, char **argv)
{
    // use getopt to get -i option -m option -o option
    int opt;
    std::string input_file, output_file, sK;
    int mode = -1;

    while ((opt = getopt(argc, argv, "i:m:o:k:")) != -1)
    {
        switch (opt)
        {
        case 'i':
            input_file = optarg;
            break;
        case 'm':
            mode = atoi(optarg);
            break;
        case 'o':
            output_file = optarg;
            break;
        case 'k':
            sK = optarg;
            break;
        default:
            std::cerr << "Usage: " << argv[0] << " -i <input_file> -m <mode> -o <output_file>" << std::endl;
            return 1;
        }
    }

    if (input_file.empty() || mode == -1 || output_file.empty())
    {
        std::cerr << "All options (-i, -m, -o) must be provided." << std::endl;
        return 1;
    }

    printf("Input file: %s\n", input_file.c_str());
    printf("Mode: %d\n", mode);
    printf("Output file: %s\n", output_file.c_str());

    QApplication a(argc, argv);

    ModelCrypto* model = new ModelCrypto;
    model->SetInputName(input_file);
    model->SetOutputName(output_file);
    model->SetK(sK);
    model->SetMode(mode);
    model->start(); // start cryptographic processing in a separate thread
    model->Wait(); // wait for the cryptographic processing to complete


    return a.exec();
}


// #include <iostream>
// #include <string>
 

// template<class TProc>
// std::string process(const std::string& sInput, 
//     const std::string& sKey, const CryptoPP::byte* aIV)
// {
//   std::string sResult;
//   TProc enc;
//   enc.SetKeyWithIV((CryptoPP::byte*)sKey.data(), sKey.size(), aIV);
 
//   CryptoPP::StringSource ss(sInput, true,
//     new CryptoPP::StreamTransformationFilter(enc,
//       new CryptoPP::StringSink(sResult)
//     )
//   );
//   return sResult;
// }
 
// std::string encrypt(const std::string& sInput, 
//     const std::string& sKey, const CryptoPP::byte* aIV)
// {
//   return process<CryptoPP::CFB_Mode<CryptoPP::AES>::Encryption>(sInput, sKey, aIV);
// }
 
// std::string decrypt(const std::string& sInput, 
//     const std::string& sKey, const CryptoPP::byte* aIV)
// {
//   return process<CryptoPP::CFB_Mode<CryptoPP::AES>::Decryption>(sInput, sKey, aIV);
// }
 
// int main()
// {
//     CryptoPP::byte aIV[CryptoPP::AES::BLOCKSIZE];
//   for (int i = 0; i < CryptoPP::AES::BLOCKSIZE; ++i)
//     aIV[i] = 0;
 
//   // key
//   std::string sKey = "<hello password>";
 
//   // initialize encryptor
//   CryptoPP::CFB_Mode<CryptoPP::AES>::Encryption enc;
//   enc.SetKeyWithIV((CryptoPP::byte*)sKey.data(), sKey.size(), aIV);
//   // content
// std::string sMyData = "Hello Crypto++";

// // encrypt
// std::string sEncData = encrypt(sMyData, sKey, aIV);
// std::cout << "Encrypted: " << sEncData << "\n";

// // decrypt
// std::string sDecData = decrypt(sEncData, sKey, aIV);
// std::cout << "Decrypted: " << sDecData << "\n";
 
// }