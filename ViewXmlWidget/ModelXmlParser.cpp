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

void ModelXmlParser::print_attributes(xmlAttr *attr)
{
    while (attr)
    {
        std::cout << "Attribute name: " << attr->name << ", value: " << attr->children->content << std::endl;
        attr = attr->next;
    }
}

// FIXME: use correct container to store parsed data (consider duplcated tags)
void ModelXmlParser::TraverseXmlTree(xmlNode *node, string path, int level, std::map<string, string> &map, ViewXmlItems* item)
{
    if (!node)
    {
        return;
    }
    xmlNode *cur_node = NULL;
    for (cur_node = node; cur_node; cur_node = cur_node->next)
    {
        if (cur_node->type == XML_ELEMENT_NODE)
        {
            string nodeName((char *)cur_node->name);
            string tag = path + "/" + nodeName;
            if (m_iVerbose)
                cout << "[TAG] " << tag << endl;

            ViewXmlItems* tagIt = NULL;
            if (item) {
                tagIt = new ViewXmlItems(item);
            } else {
                tagIt = new ViewXmlItems(twContainer);
            }

            tagIt->setText(0, nodeName.c_str());

            map[tag] = "";

            // cout << "Parent node:" << path << endl;

            // cout << "Child node column 0: " << cur_node->name << endl;

            xmlAttr *attr = cur_node->properties;
            string tagAttr = tag;
            while (attr)
            {
                tagAttr = tag + "/" + string((char *)attr->name);
                string val((char *)attr->children->content);

                // cout << "Child node column 1: " << attr->name << endl;
                string arrName = string((char *)attr->name);
                tagIt->setText(1, arrName.c_str());

                map[tagAttr] = val;
                // cout << "Child node column 2: " << val << endl;
                tagIt->setText(2, val.c_str());

                if (m_iVerbose)
                    cout << "[TAG/ATTR] " << tagAttr << ", [VALUE] " << val << std::endl;
                attr = attr->next;
            }

            TraverseXmlTree(cur_node->children, tag, level + 1, map, tagIt);
        }
        else if (cur_node->type == XML_TEXT_NODE && cur_node->content)
        {
            // check content empty or not;
            string ct((char *)cur_node->content);
            QString qs((char *)cur_node->content);
            qs = qs.simplified();
            string value = qs.toStdString();
            string tagAttrContent = path + "/content";

            cout << "child node column 3: " << value << endl;

            if (qs.length() > 0)
            {

                map[tagAttrContent] = value;
                if (m_iVerbose)
                    cout << "[TAG/ATTR/CONTENT] " << tagAttrContent << " [Content]: " << value << endl;
            }
            else
            {
                map[tagAttrContent] = "";
            }

            if (item) {
                item->setText(3, value.c_str());
            }
        }
    }
}

void ModelXmlParser::PrintNodeInfo(xmlNode *node)
{
    if (node == NULL)
    {
        return;
    }

    switch (node->type)
    {

    case XML_ELEMENT_NODE:
    {
        cout << "[Node type: ELEMENT] name: " << node->name << endl;
        break;
    }
    case XML_TEXT_NODE:
    {
        cout << "[Node type: TEXT] content: " << node->content << endl;
        break;
    }
    case XML_CDATA_SECTION_NODE:
    {
        cout << "[Node type: CDATA] content: " << node->content << endl;
        break;
    }
    case XML_COMMENT_NODE:
    {
        cout << "[Node type: COMMENT] content: " << node->content << endl;
        break;
    }
    case XML_PI_NODE:
    {
        cout << "[Node type: PI] target: " << node->name << ", content: " << node->content << endl;
        break;
    }
    case XML_DOCUMENT_NODE:
    {
        cout << "[Node type: DOCUMENT] content: " << node->content << endl;
        break;
    }
    case XML_DTD_NODE:
    {
        cout << "[Node type: DTD] name: " << node->name << ", content: " << node->content << endl;
        break;
    }
    case XML_ENTITY_NODE:
    {
        cout << "[Node type: ENTITY] name: " << node->name << ", content: " << node->content << endl;
        break;
    }
    case XML_ATTRIBUTE_NODE:
    {
        cout << "[Node type: ATTRIBUTE] name: " << node->name << ", content: " << node->content << endl;
        break;
    }

    default:
        break;
    }

    print_attributes(node->properties);
}

void ModelXmlParser::TraverseXmlTree(xmlNode *node, std::function<void(xmlNode *)> callback)
{
    if (node == NULL)
    {
        return;
    }

    callback(node); // Call callback for the current node

    for (xmlNode *cur_node = node->children; cur_node; cur_node = cur_node->next)
    {
        TraverseXmlTree(cur_node, callback);
    }
}

void ModelXmlParser::run()
{
    printf("libxml version: %s\n", LIBXML_DOTTED_VERSION);

    m_xmlDoc = xmlReadFile(m_sFname.c_str(), NULL, 0);
    if (!m_xmlDoc)
    {
        cerr << "Failed to parse XML file: " << m_sFname.c_str() << endl;
        return;
    }

    xmlNode *root = xmlDocGetRootElement(m_xmlDoc);
    if (!root)
    {
        cerr << "No root element in XML file: " << m_sFname.c_str() << endl;
        xmlFreeDoc(m_xmlDoc);
        return;
    }

    // const auto cb = [this](xmlNode *node)
    // {
    //     PrintNodeInfo(node);
    // };
    string path = "";
    int level = 0;
    std::map<string, string> map;
    TraverseXmlTree(root, path, level, map);

    // print map
    // for(auto& it : map) {
    //     std::cout << it.first << ": " << it.second << std::endl;
    // }

    emit AllPageReaded();
}