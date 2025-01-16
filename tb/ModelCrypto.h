#ifndef MODEL_CRYPTO_H
#define MODEL_CRYPTO_H

#include <iostream>
#include <string>
#include <fstream>
#include <QApplication>
#include <QThread>

#include <cryptopp/cryptlib.h>
#include <cryptopp/filters.h>

#include <cryptopp/modes.h>
#include <cryptopp/aes.h>

using namespace std;

class ModelCrypto : public QThread
{
    Q_OBJECT
public:
    ModelCrypto(QObject *parent = nullptr);
    ~ModelCrypto() = default;
    void SetInputName(const string &inputName) { m_sIptName = inputName; }
    void SetOutputName(const string &outputName) { m_sOptName = outputName; }
    void SetK(const string &k) { m_sK = k; }
    void SetMode(int mode) { m_iMode = mode; }
    void Wait();

protected:
    void run() override;

private:
    string m_sIptName;
    string m_sOptName;
    string m_sK;
    int m_iMode;
};

#endif /* MODEL_CRYPTO_H */