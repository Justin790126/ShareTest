#include "ModelCrypto.h"

ModelCrypto::ModelCrypto(QObject *parent)
    : QThread(parent)
{
}

void ModelCrypto::Wait()
{
    while (isRunning())
    {
        usleep(1000);
        QApplication::processEvents();
    }
}

template<class TProc>
std::string process(const std::string& sInput, 
    const std::string& sKey, const CryptoPP::byte* aIV)
{
  std::string sResult;
  TProc enc;
  enc.SetKeyWithIV((CryptoPP::byte*)sKey.data(), sKey.size(), aIV);
 
  CryptoPP::StringSource ss(sInput, true,
    new CryptoPP::StreamTransformationFilter(enc,
      new CryptoPP::StringSink(sResult)
    )
  );
  return sResult;
}
 
std::string encrypt(const std::string& sInput, 
    const std::string& sKey, const CryptoPP::byte* aIV)
{
  return process<CryptoPP::CFB_Mode<CryptoPP::AES>::Encryption>(sInput, sKey, aIV);
}
 
std::string decrypt(const std::string& sInput, 
    const std::string& sKey, const CryptoPP::byte* aIV)
{
  return process<CryptoPP::CFB_Mode<CryptoPP::AES>::Decryption>(sInput, sKey, aIV);
}
 
void ModelCrypto::run()
{
    CryptoPP::byte aIV[CryptoPP::AES::BLOCKSIZE];
    // cout << "Number of passwd: " << (int)CryptoPP::AES::BLOCKSIZE << endl;
    // for (int i = 0; i < CryptoPP::AES::BLOCKSIZE; ++i)
    //     aIV[i] = 0;

    // initialize encryptor
    CryptoPP::CFB_Mode<CryptoPP::AES>::Encryption enc;
    enc.SetKeyWithIV((CryptoPP::byte *)m_sK.data(), m_sK.size(), aIV);

    if (m_iMode == 1)
    {
        // encrypt mode with m_sK as key

        string fileContent;
        // read file content
        ifstream inputFile(m_sIptName);
        inputFile.seekg(0, std::ios::end);
        fileContent.resize(inputFile.tellg());
        inputFile.seekg(0);
        inputFile.read(&fileContent[0], fileContent.size());
        inputFile.close();

        std::string sEncData = encrypt(fileContent, m_sK, aIV);
        // std::cout << "Encrypted: " << sEncData << "\n";

        ofstream outputFile(m_sOptName);
        outputFile.write(sEncData.c_str(), sEncData.size());
        outputFile.close();
    }
    else if (m_iMode == 0)
    {
        // decrypt mode with m_sK as key

        string fileContent;
        // read file content
        ifstream inputFile(m_sIptName);
        inputFile.seekg(0, std::ios::end);
        fileContent.resize(inputFile.tellg());
        inputFile.seekg(0);
        inputFile.read(&fileContent[0], fileContent.size());
        inputFile.close();

        std::string sDecData = decrypt(fileContent, m_sK, aIV);
        // std::cout << "Decrypted: " << sDecData << "\n";

        ofstream outputFile(m_sOptName);
        outputFile.write(sDecData.c_str(), sDecData.size());
        outputFile.close();
    }
}