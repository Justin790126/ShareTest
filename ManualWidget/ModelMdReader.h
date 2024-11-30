#ifndef MODEL_MD_READER_H
#define MODEL_MD_READER_H

#include <QThread>
#include <iostream>
#include <fstream>
#include <regex>
#include "md2html.h"
#include <sstream>

using namespace std;

class MdNode
{
public:
    MdNode() {};
    MdNode(string key, string url, int lvl) : m_sKey(key), m_sUrl(url), m_iLevel(lvl) {}
    void SetKey(string key) { m_sKey = key; }
    void SetUrl(string url) { m_sUrl = url; }
    void SetLevel(int lvl) { m_iLevel = lvl; }
    int GetLevel() { return m_iLevel; }
    string GetKey() { return m_sKey; }
    string GetUrl() { return m_sUrl; }
    void AddChild(MdNode *node) { m_vsChildren.push_back(node); }
    vector<MdNode *> *GetChildren() { return &m_vsChildren; }
    void SetParent(MdNode *parent) { m_sParent = parent; }
    MdNode *GetParent() { return m_sParent; }
    void SetHtmlContent(const std::string &htmlContent) { m_sHtmlContent = htmlContent; }
    string GetHtmlContent() { return m_sHtmlContent; }

    friend ostream &operator<<(ostream &os, const MdNode &node)
    {
        os << "Key: " << node.m_sKey << ", Url: " << node.m_sUrl << ", Level: " << node.m_iLevel << endl;
        return os;
    }

private:
    string m_sKey;
    string m_sUrl;
    string m_sHtmlContent;
    int m_iLevel;
    MdNode *m_sParent = NULL;
    vector<MdNode *> m_vsChildren;
};

class SearchInfo
{
public:
    SearchInfo() {}
    SearchInfo(const string &key, const string &url, const string &resultLine, int lineNum) : m_sKey(key), m_sUrl(url), m_sResultLine(resultLine), m_iLineNum(lineNum)
    {
    }

    string GetInfo()
    {
        char buf[1024];
        sprintf(buf, "%s found at line %d in [%s](%s)", m_sResultLine.c_str(), m_iLineNum, m_sKey.c_str(), m_sUrl.c_str());
        return buf;
    }
    bool compInfo(const string& info) {
        string srcInfo = GetInfo();
        return strcmp(srcInfo.c_str(), info.c_str()) == 0;
    }
    
    void GetKeyUrlFromInfo(const string& iInfo, string &key, string &url) {
        std::regex link_regex(R"(\[([^\]]+)\]\(([^)]+)\))");
        std::smatch match;
        if (std::regex_search(iInfo, match, link_regex)) {
            key = match[1];
            url = match[2];
        }
        else {
            key = "";
            url = "";
        }
    }

    void SetNode(MdNode* node) { m_sNode = node; }
    MdNode* GetNode() { return m_sNode; }

    friend ostream &operator<<(ostream &os, const SearchInfo &info)
    {
        os << "[Search Result] Key: " << info.m_sKey << ", Url: " << info.m_sUrl << ", Result Line: " << info.m_sResultLine << ", Line Number: " << info.m_iLineNum << endl;
        return os;
    }

private:
    string m_sKey;
    string m_sUrl;
    string m_sResultLine;
    int m_iLineNum;
    MdNode* m_sNode=NULL;
};

class ModelMdReader : public QThread
{
    Q_OBJECT

public:
    explicit ModelMdReader(QObject *parent = nullptr);
    void ParseMdRec(string fname, string folder, int level, MdNode *node);

    void run() override;

    MdNode *GetRoot() { return m_sRoot; }
    int GetMaxLevel() { return m_iMaxLevel; }
    void TraverseMdNode(MdNode *node, std::function<void(MdNode *)> callback);
    void SetFname(const std::string &fname) { m_sFname = fname; }

    void testRun();

    void Search(const std::string &key);

    vector<SearchInfo> *GetSearchInfos() { return &m_vSearchInfos; }
signals:
    void allReaded();

private:
    bool isImageFile(const std::string &url);
    string m_sFname;
    MdNode *m_sRoot = NULL;
    int m_iMaxLevel = 0;

    vector<SearchInfo> m_vSearchInfos;
};

#endif /* MODEL_MD_READER_H */