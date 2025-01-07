#include "ModelOvlConf.h"

ModelOvlConf::ModelOvlConf()
{
    m_iVerbose = 0;
}

ModelOvlConf::~ModelOvlConf()
{
    m_vNameAndInfo.clear();
}

void ModelOvlConf::Wait()
{
    while (this->isRunning())
    {
        usleep(1000);
        QApplication::processEvents();
    }
    
}

OvlProductInfo* ModelOvlConf::AddNewProductInfo(string& pdName, double& dieW, double& dieH, double& dieOffsetX, double& dieOffsetY)
{
    OvlProductInfo* info = new OvlProductInfo;
    info->SetProductName(pdName);
    info->SetDieWidth(dieW);
    info->SetDieHeight(dieH);
    info->SetDieOffset(dieOffsetX, dieOffsetY);
    m_vNameAndInfo.push_back(*info);
    return info;
}

void ModelOvlConf::ReadOvlConfig()
{
    cout << "parsing rcsv table..." << endl;

    // use if stream to open file
    ifstream fp(m_sFname);
    if (!fp.is_open())
    {
        return;
    }

    string line;
    m_vNameAndInfo.clear();
    while (getline(fp, line))
    {
        // separate line by space
        vector<string> tokens;
        tokens.reserve(5);
        istringstream iss(line);
        string token;
        while (getline(iss, token, ' '))
        {
            tokens.push_back(token);
        }

        if (tokens.size() == 5)
        {
            string pdName = tokens[0];
            double wfLen = strtod(tokens[1].c_str(), NULL);
            double wfSize = strtod(tokens[2].c_str(), NULL);
            double wfOffsetX = strtod(tokens[3].c_str(), NULL);
            double wfOffsetY = strtod(tokens[4].c_str(), NULL);

            OvlProductInfo info;
            info.SetProductName(pdName);
            info.SetDieWidth(wfLen);
            info.SetDieHeight(wfSize);
            info.SetDieOffset(wfOffsetX, wfOffsetY);
            m_vNameAndInfo.emplace_back(std::move(info));
        }
    }

    // iterate m_vNameAndInfo
    if (m_iVerbose)
    {
        for (const auto &pair : m_vNameAndInfo)
        {
            cout << pair << endl;
        }
    }

    fp.close();

    emit allPageReaded();
}

void ModelOvlConf::WriteOvlConfig()
{
    cout << "writing rcsv table..." << endl;

    // use ofstream to open file
    ofstream fp(m_sFname);
    if (!fp.is_open())
    {
        return;
    }

    // iterate m_vNameAndInfo by index
    for (size_t i = 0; i < m_vNameAndInfo.size(); i++) {
        OvlProductInfo *pInfo = &(m_vNameAndInfo[i]);
        fp << pInfo->GetProductName() << " "
           << pInfo->GetDieWidth() << " "
           << pInfo->GetDieHeight() << " "
           << pInfo->GetDieOffsetX() << " "
           << pInfo->GetDieOffsetY() << endl;
    }

    fp.close();

    emit allPageWritten();
}

void ModelOvlConf::run()
{
    if (m_iOvlCfgMode == (int)OVL_READ_CFG)
    {

        ReadOvlConfig();
    }
    else if (m_iOvlCfgMode == (int)OVL_WRITE_CFG)
    {

        WriteOvlConfig();
    }
}