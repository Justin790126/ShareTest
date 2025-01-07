#ifndef MODEL_OVL_CONF_H
#define MODEL_OVL_CONF_H

#include <iostream>
#include <map>
#include <fstream>
#include <sstream>
#include <vector>
using namespace std;

#include <QThread>
#include <QApplication>

class OvlProductInfo
{
public:
    OvlProductInfo() {}
    ~OvlProductInfo() {}

    void SetProductName(string &name) { m_sProductName = name; }
    string GetProductName() const { return m_sProductName; }
    void SetDieWidth(double wfLen) { m_dDieW = wfLen; }
    double GetDieWidth() const { return m_dDieW; }
    void SetDieHeight(double wfSize) { m_dDieH = wfSize; }
    double GetDieHeight() const { return m_dDieH; }
    void SetDieOffset(double x, double y)
    {
        m_dDieOffsetX = x;
        m_dDieOffsetY = y;
    }
    double GetDieOffsetX() { return m_dDieOffsetX; }
    double GetDieOffsetY() { return m_dDieOffsetY; }

    // overload cout
    friend ostream &operator<<(ostream &os, const OvlProductInfo &info)
    {
        os << "ProductName: " << info.m_sProductName << ", DieW: " << info.m_dDieW
           << ", DieH: " << info.m_dDieH << ", Die offsetX: " << info.m_dDieOffsetX
           << ", Die offsetY: " << info.m_dDieOffsetY << endl;
        return os;
    }

private:
    string m_sProductName;
    double m_dDieW;
    double m_dDieH;
    double m_dDieOffsetX, m_dDieOffsetY;
};

enum OvlConfMode
{
    OVL_READ_CFG,
    OVL_WRITE_CFG
};

class ModelOvlConf : public QThread
{
    Q_OBJECT
public:
    ModelOvlConf();
    ~ModelOvlConf();

    void SetFname(const string fname) { m_sFname = fname; }
    string GetFname() const { return m_sFname; }
    vector<OvlProductInfo> *GetProductInfo() { return &m_vNameAndInfo; }
    OvlProductInfo* AddNewProductInfo(string& pdName, double& dieW, double& dieH, double& dieOffsetX, double& dieOffsetY);
    void SetProductInfo(vector<OvlProductInfo>& vInfo) { m_vNameAndInfo = std::move(vInfo); }
    void SetWorkerMode(int mode) { m_iOvlCfgMode = mode; }
    void Wait();
signals:
    void allPageReaded();
    void allPageWritten();

protected:
    virtual void run() override;

private:
    void ReadOvlConfig();
    void WriteOvlConfig();

private:
    int m_iVerbose;
    int m_iOvlCfgMode = 0;
    string m_sFname;

    vector<OvlProductInfo> m_vNameAndInfo;
};

#endif /* MODEL_OVL_CONF_H */