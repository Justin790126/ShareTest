#ifndef MODEL_NPZ_H
#define MODEL_NPZ_H

#include <QThread>
#include <iostream>
#include <string>
#include <fstream>
#include <stdio.h>
#include <map>

using namespace std;

class ModelNpz : public QThread
{
    Q_OBJECT
public:
    explicit ModelNpz(QObject *parent = nullptr)
        : QThread(parent) {}
    ~ModelNpz() = default;

    void SetFileName(const string &sFname) { m_sFname = sFname; }
    const string &GetFileName() const { return m_sFname; }

protected:
    virtual void run() override;

    string m_sFname;

    ifstream* m_fp = NULL;

private:
    string m_sNpVer;
    std::map<string, string> m_NpzHeader;


    const string m_sMagic = "NUMPY";
    const unsigned char m_ucMagicByte = 0x93;
    const string m_sDescKey = "descr";
    const string m_sFortranKey = "fortran_order";
    const string m_sShapeKey = "shape";
};

#endif /* MODEL_NPZ_H */