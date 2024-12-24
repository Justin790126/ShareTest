#ifndef MODEL_XML_PARSER_H
#define MODEL_XML_PARSER_H

#include <QThread>
#include <iostream>
#include <libxml/xmlmemory.h>
#include <libxml/parser.h>
#include <libxml/tree.h>
#include <functional>
#include <map>
#include <vector>
#include "ViewXmlItems.h"

using namespace std;

enum XmlWokerMode
{
    MODE_CREATE_ITEMS,
    MODE_COMPARE_TWO_FILES
};

struct XmlDiff {
    string path;
    string node1;
    string node2;
    string message;
};

class ModelXmlParser : public QThread
{
    Q_OBJECT
public:
    ModelXmlParser(QObject *parent = NULL);
    ~ModelXmlParser();
    void SetFileName(const string &fname) { m_sFname1 = fname; }
    string GetFileName() const { return m_sFname1; }
    void SetFileName2(const string &fname) { m_sFname2 = fname; }
    string GetFileName2() const { return m_sFname2; }

    void TraverseXmlTree(xmlNode *node, string path, int level, std::map<string, ViewXmlItems*> &map, ViewXmlItems* item=NULL);
    void TraverseXmlTree(xmlNode *node, string path, int level, std::map<string, int>& map);


    // Write TraverseXmlTree that accept std::function<void(xmlNode*)> callback;
    void TraverseXmlTree(xmlNode *node, std::function<void(xmlNode *)> callback);

    void print_attributes(xmlAttr *attr);

    void PrintNodeInfo(xmlNode *node);

    void SetTreeWidget(QTreeWidget* tw) { twContainer = tw; }
    QTreeWidget* GetTreeWidget() const { return twContainer; }

    void SetWorkerMode(int mode) { m_iWorkerMode = mode; }
    int GetWorkerMode() const { return m_iWorkerMode; }

    string getNodePath(xmlNodePtr node);
    void CompareNodes(xmlNodePtr node1, xmlNodePtr node2, const string& path);

signals:
    void AllPageReaded();

private:
    virtual void run() override;

    string m_sFname1;
    string m_sFname2;

    xmlDoc *m_xmlDoc = NULL;

    int m_iVerbose = 1;

    std::map<string, ViewXmlItems *> m_mKeyItems;
    std::map<string, int> m_mKeyStatistics;
    QTreeWidget* twContainer = NULL;

    int m_iWorkerMode = 0;

    void CreateXmlItems();

    void CompareTwoFiles();
};

#endif /* MODEL_XML_PARSER_H */