#ifndef MODEL_XML_PARSER_H
#define MODEL_XML_PARSER_H

#include <QThread>
#include <iostream>
#include <libxml/parser.h>
#include <libxml/tree.h>
#include <functional>


using namespace std;

class ModelXmlParser : public QThread
{
    Q_OBJECT
    public:
        ModelXmlParser(QObject *parent=NULL);
        ~ModelXmlParser();
        void SetFileName(const string& fname) { m_sFname = fname; }
        string GetFileName() const { return m_sFname; }

        void TraverseXmlTree(xmlNode* node);

        // Write TraverseXmlTree that accept std::function<void(xmlNode*)> callback;
        void TraverseXmlTree(xmlNode* node, std::function<void(xmlNode*)> callback);

        void print_attributes(xmlAttr *attr);

    signals:
        void AllPageReaded();
    
    private:
        virtual void run() override;

        string m_sFname;

        xmlDoc* m_xmlDoc = NULL;
};


#endif /* MODEL_XML_PARSER_H */