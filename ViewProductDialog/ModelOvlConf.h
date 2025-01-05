#ifndef MODEL_OVL_CONF_H
#define MODEL_OVL_CONF_H

#include <iostream>
#include <map>
#include <fstream>
#include <sstream>
#include <vector>
using namespace std;

#include <QThread>

class OvlProductInfo
{
public:
    OvlProductInfo() {}
    ~OvlProductInfo() {}

    void SetProductName(string &name) { m_sProductName = name; }
    string GetProductName() const { return m_sProductName; }
    void SetWfLen(double wfLen) { m_dWfLen = wfLen; }
    double GetWfLen() const { return m_dWfLen; }
    void SetWfSize(double wfSize) { m_dWfSize = wfSize; }
    double GetWfSize() const { return m_dWfSize; }
    void SetWfOffset(double x, double y)
    {
        m_dWfOffsetX = x;
        m_dWfOffsetY = y;
    }
    double GetWfOffsetX() { return m_dWfOffsetX; }
    double GetWfOffsetY() { return m_dWfOffsetY; }

    // overload cout
    friend ostream &operator<<(ostream &os, const OvlProductInfo &info)
    {
        os << "ProductName: " << info.m_sProductName << ", WfLen: " << info.m_dWfLen
           << ", WfSize: " << info.m_dWfSize << ", WfOffsetX: " << info.m_dWfOffsetX
           << ", WfOffsetY: " << info.m_dWfOffsetY << endl;
        return os;
    }

private:
    string m_sProductName;
    double m_dWfLen;
    double m_dWfSize;
    double m_dWfOffsetX, m_dWfOffsetY;
};

class ModelOvlConf : public QThread
{
    Q_OBJECT
public:
    ModelOvlConf();
    ~ModelOvlConf();

    void SetFname(const string fname) { m_sFname = fname; }
    string GetFname() const { return m_sFname; }
    std::map<string, OvlProductInfo> *GetProductInfo() { return &m_mNameAndInfo; }

signals:
    void allPageReaded();

protected:
    virtual void run() override;

private:
    int m_iVerbose;
    string m_sFname;

    map<string, OvlProductInfo> m_mNameAndInfo;
};

#endif /* MODEL_OVL_CONF_H */