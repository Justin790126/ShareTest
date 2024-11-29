#ifndef MODEL_MD_READER_H
#define MODEL_MD_READER_H

#include <QThread>
#include <iostream>
#include <fstream>
#include <regex>
#include "md2html.h"
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
    MdNode* m_sParent = NULL;
    vector<MdNode *> m_vsChildren;
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
signals:
    void allReaded();

private:
    bool isImageFile(const std::string &url);
    string m_sFname;
    MdNode *m_sRoot = NULL;
    int m_iMaxLevel = 0;
};

#endif /* MODEL_MD_READER_H */