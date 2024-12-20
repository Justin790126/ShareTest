#include "ModelXmlParser.h"

ModelXmlParser::ModelXmlParser(QObject *parent) : QThread(parent)
{

}

ModelXmlParser::~ModelXmlParser()
{
    // Clear all data
    xmlFreeDoc(m_xmlDoc);
    xmlCleanupParser();
}

void ModelXmlParser::print_attributes(xmlAttr *attr) {
    while (attr) {
        std::cout << "Attribute name: " << attr->name << ", value: " << attr->children->content << std::endl;
        attr = attr->next;
    }
}

void ModelXmlParser::TraverseXmlTree(xmlNode* node)
{
    xmlNode* cur_node = NULL;
    for (cur_node = node; cur_node; cur_node = cur_node->next) {
        if (cur_node->type == XML_ELEMENT_NODE) {
            cout << "[Node name]: " << cur_node->name << endl;

            print_attributes(cur_node->properties);

            TraverseXmlTree(cur_node->children);
        } else if (cur_node->type == XML_TEXT_NODE && cur_node->content) {
            cout << "[Node type: TEXT] content: " << cur_node->content << endl;
        }
    }
}

void ModelXmlParser::TraverseXmlTree(xmlNode* node, std::function<void(xmlNode*)> callback)
{
    xmlNode* cur_node = NULL;
    for (cur_node = node; cur_node; cur_node = cur_node->next) {
        if (cur_node->type == XML_ELEMENT_NODE) {
            callback(cur_node);
            TraverseXmlTree(cur_node->children, callback);
        } else if (cur_node->type == XML_TEXT_NODE && cur_node->content) {
            callback(cur_node);
        }
    }
}

void ModelXmlParser::run()
{
    cout << "start parsing XML file" << endl;
    printf("libxml version: %s\n", LIBXML_DOTTED_VERSION);

    m_xmlDoc = xmlReadFile(m_sFname.c_str(), NULL, 0);
    if (!m_xmlDoc) {
        cerr << "Failed to parse XML file: " << m_sFname.c_str() << endl;
        return;
    }

    xmlNode* root = xmlDocGetRootElement(m_xmlDoc);
    if (!root) {
        cerr << "No root element in XML file: " << m_sFname.c_str() << endl;
        xmlFreeDoc(m_xmlDoc);
        return;
    }

    const auto cb = [this](xmlNode* node) {
        // print
        cout << "[Node name]: " << node->name << endl;
        print_attributes(node->properties);
    };
    TraverseXmlTree(root, cb);

    

    emit AllPageReaded();
}