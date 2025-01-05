#include "ModelOvlConf.h"

ModelOvlConf::ModelOvlConf()
{
    m_iVerbose = 0;
}

ModelOvlConf::~ModelOvlConf()
{
    m_mNameAndInfo.clear();
}

void ModelOvlConf::run()
{
    cout << "parsing rcsv table..." << endl;

    // use if stream to open file
    ifstream fp(m_sFname);
    if (!fp.is_open())
    {
        return;
    }

    string line;
    m_mNameAndInfo.clear();
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
            info.SetWfLen(wfLen);
            info.SetWfSize(wfSize);
            info.SetWfOffset(wfOffsetX, wfOffsetY);
            m_mNameAndInfo[pdName] = std::move(info);
        }
    }

    // iterate m_mNameAndInfo
    if (m_iVerbose)
    {
        for (const auto &pair : m_mNameAndInfo)
        {
            cout << pair.first << " " << pair.second << endl;
        }
    }

    fp.close();

    emit allPageReaded();
}