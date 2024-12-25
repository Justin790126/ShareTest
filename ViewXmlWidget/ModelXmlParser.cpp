#include "ModelXmlParser.h"

ModelXmlParser::ModelXmlParser(QObject *parent) : QThread(parent)
{
}

ModelXmlParser::~ModelXmlParser()
{
    // Clear all data
    xmlFreeDoc(m_xmlDoc1);
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
void ModelXmlParser::TraverseXmlTree(xmlNode *node, string path, int level, std::map<string, ViewXmlItems *> &map, ViewXmlItems *item, QTreeWidget* twContainer)
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

            ViewXmlItems *tagIt = NULL;
            if (item)
            {
                tagIt = new ViewXmlItems(item);
            }
            else
            {
                tagIt = new ViewXmlItems(twContainer);
            }
            tagIt->SetMapKey(tag);
            tagIt->setText(0, nodeName.c_str());

            map[tag] = tagIt;

            xmlAttr *attr = cur_node->properties;
            string tagAttr = tag;
            ViewXmlItems *attrIt = NULL;
            while (attr)
            {

                string val((char *)attr->children->content);
                string arrName = string((char *)attr->name);
                tagAttr = tag + "/" + arrName + "/" + val;
                
                
                attrIt = new ViewXmlItems(tagIt);
                attrIt->setText(2, arrName.c_str());
                attrIt->setText(3, val.c_str());
                attrIt->SetMapKey(tagAttr);
                attrIt->SetAttrValue(val);
                map[tagAttr] = attrIt;

                if (m_iVerbose)
                    cout << "[TAG/ATTR] " << tagAttr << ", [VALUE] " << val << std::endl;
                attr = attr->next;
            }

            TraverseXmlTree(cur_node->children, tag, level + 1, map, tagIt, twContainer);
        }
        else if (cur_node->type == XML_TEXT_NODE && cur_node->content)
        {
            // check content empty or not;
            string ct((char *)cur_node->content);
            QString qs((char *)cur_node->content);
            qs = qs.simplified();
            string value = qs.toStdString();
            string tagAttrContent = path + "/text/" + value;
            if (m_iVerbose)
                cout << "[TAG/ATTR/CONTENT] " << tagAttrContent << " [Content]: " << value << endl;
            if (item)
            {
                item->SetHasContent(qs.length() > 0);
                item->SetContent(value);
                item->setText(1, value.c_str());
            }
            map[tagAttrContent] = item;
        }
    }
}

void ModelXmlParser::TraverseXmlTree(xmlNode *node, string path, int level, std::map<string, int> &map)
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

            if (map.count(tag) == 0) {
                map[tag] = 1;
            } else {
                map[tag]++;
            }

            xmlAttr *attr = cur_node->properties;
            string tagAttr = tag;
            
            while (attr)
            {

                string val((char *)attr->children->content);
                string arrName = string((char *)attr->name);
                tagAttr = tag + "/" + arrName + "/" + val;

                if (map.count(tagAttr) == 0) {
                    map[tagAttr] = 1;
                } else {
                    map[tagAttr]++;
                }


                if (m_iVerbose)
                    cout << "[TAG/ATTR/Value] " << tagAttr << std::endl;
                attr = attr->next;
            }

            TraverseXmlTree(cur_node->children, tag, level + 1, map);
        }
        else if (cur_node->type == XML_TEXT_NODE && cur_node->content)
        {
            // check content empty or not;
            string ct((char *)cur_node->content);
            QString qs((char *)cur_node->content);
            qs = qs.simplified();
            string value = qs.toStdString();
            string tagAttrContent = path + "/text/" + value;
            if (map.count(tagAttrContent) == 0)
            {
                map[tagAttrContent] = 1;
            } else {
                map[tagAttrContent]++;
            }

            if (m_iVerbose)
                cout << "[TAG/ATTR/text/value] " << tagAttrContent << endl;
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


void ModelXmlParser::CreateXmlItems()
{
    m_xmlDoc1 = xmlReadFile(m_sFname1.c_str(), NULL, 0);
    if (!m_xmlDoc1)
    {
        cerr << "Failed to parse XML file: " << m_sFname1.c_str() << endl;
        return;
    }

    xmlNode *root = xmlDocGetRootElement(m_xmlDoc1);
    if (!root)
    {
        cerr << "No root element in XML file: " << m_sFname1.c_str() << endl;
        xmlFreeDoc(m_xmlDoc1);
        return;
    }

    string path = "";
    int level = 0;
    m_mKeyItems1.clear();
    TraverseXmlTree(root, path, level, m_mKeyItems1, NULL, twContainer1);

    emit AllPageReaded(twContainer1);
}

string ModelXmlParser::getNodePath(xmlNodePtr node)
{
    string path;
    while (node)
    {
        path = "/" + string((char *)node->name) + path;
        node = node->parent;
    }
    return path;
}
void ModelXmlParser::CompareNodes(xmlNodePtr node1, xmlNodePtr node2, const string &path)
{
    if (node1 == NULL && node2 == NULL)
    {
        return; // Both nodes are NULL
    }
    if (node1 == NULL || node2 == NULL)
    {
        cout << "Node mismatch at path: " << path << endl;
        return; // Only one node is NULL
    }

    // Compare node types
    if (node1->type != node2->type)
    {
        cout << "Node type mismatch at path: " << path << endl;
        return;
    }

    if (node1->type == XML_ELEMENT_NODE && node2->type == XML_ELEMENT_NODE)
    {
        // Compare node names
        if (xmlStrcmp(node1->name, node2->name) != 0)
        {
            cout << "Node name mismatch at path: " << path << " - " << (const char *)node1->name << " vs " << (const char *)node2->name << endl;
            return;
        }
        // Compare node attributes
        xmlAttrPtr attr1 = node1->properties;
        xmlAttrPtr attr2 = node2->properties;

        while (attr1 != NULL && attr2 != NULL)
        {
            if (xmlStrcmp(attr1->name, attr2->name) != 0)
            {
                cout << "Attribute name mismatch for node '" << (const char *)node1->name << "' at path: " << path << endl;
            }
            if (xmlStrcmp(attr1->children->content, attr2->children->content) != 0)
            {
                cout << "Attribute value mismatch for node '" << (const char *)node1->name << "' and attribute '" << (const char *)attr1->name << "' at path: " << path << endl;
            }
            attr1 = attr1->next;
            attr2 = attr2->next;
        }

        // If one node has attributes and the other doesn't
        if ((attr1 != NULL && attr2 == NULL) || (attr1 == NULL && attr2 != NULL))
        {
            cout << "Attribute count mismatch for node '" << (const char *)node1->name << "' at path: " << path << endl;
            return;
        }

        // Compare child nodes recursively
        xmlNodePtr child1 = node1->children;
        xmlNodePtr child2 = node2->children;
        while (child1 != NULL && child2 != NULL)
        {
            CompareNodes(child1, child2, path + "/" + (const char *)child1->name);
            child1 = child1->next;
            child2 = child2->next;
        }

        // // If one node has children and the other doesn't
        if ((child1 != NULL && child2 == NULL) || (child1 == NULL && child2 != NULL))
        {
            cout << "Child node count mismatch for node '" << (const char *)node1->name << "' at path: " << path << endl;
            return;
        }
    }
    else if (node1->type == XML_TEXT_NODE && node1->content &&
             node2->type == XML_TEXT_NODE && node2->content)
    {
        if (xmlStrcmp(node1->content, node2->content) != 0)
        {
            cout << "Text content mismatch at path: " << path << endl;
        }
    }
}
void ModelXmlParser::CompareTwoFiles()
{
    m_xmlDoc1 = xmlReadFile(m_sFname1.c_str(), NULL, 0);
    if (!m_xmlDoc1)
    {
        cerr << "Failed to parse XML file: " << m_sFname1.c_str() << endl;
        return;
    }

    xmlNode *root1 = xmlDocGetRootElement(m_xmlDoc1);
    if (!root1)
    {
        cerr << "No root element in XML file: " << m_sFname1.c_str() << endl;
        xmlFreeDoc(m_xmlDoc1);
        return;
    }

    string path = "";
    int level = 0;
    m_mKeyStatistics1.clear();
    TraverseXmlTree(root1, path, level, m_mKeyStatistics1);
    m_mKeyItems1.clear();
    TraverseXmlTree(root1, path, level, m_mKeyItems1, NULL, twContainer1);
    emit AllPageReaded(twContainer1);
    // iterate m_mKeyStatistics1
    for (auto &item : m_mKeyStatistics1)
    {
        cout << item.first << ": " << item.second << endl;
    }

    m_xmlDoc2 = xmlReadFile(m_sFname2.c_str(), NULL, 0);
    if (!m_xmlDoc2) {
        cerr << "Failed to parse XML file: " << m_sFname2.c_str() << endl;
        xmlFreeDoc(m_xmlDoc1);
        xmlFreeDoc(m_xmlDoc2);
        return;
    }

    xmlNode *root2 = xmlDocGetRootElement(m_xmlDoc2);
    if (!root2) {
        cerr << "No root element in XML file: " << m_sFname2.c_str() << endl;
        xmlFreeDoc(m_xmlDoc1);
        xmlFreeDoc(m_xmlDoc2);
        return;
    }


    string path2 = "";
    int level2 = 0;
    m_mKeyStatistics2.clear();
    TraverseXmlTree(root2, path2, level2, m_mKeyStatistics2);
    m_mKeyItems2.clear();
    TraverseXmlTree(root1, path, level, m_mKeyItems2, NULL, twContainer2);

    emit AllPageReaded(twContainer2);
    // iterate m_mKeyStatistics2
    for (auto &item : m_mKeyStatistics2)
    {
        cout << item.first << ": " << item.second << endl;
    }

    // compare m_mKeyStatistics1 and m_mKeyStatistics2, then get the items from m_mKeyItems1 and m_mKeyItems2
    for (auto &item1 : m_mKeyStatistics1) {
        auto it2 = m_mKeyStatistics2.find(item1.first);
        if (it2!= m_mKeyStatistics2.end()) {
            if (it2->second!= item1.second) {
                cout << "Key '" << item1.first << "' has different counts in both files: " << item1.second << " vs " << it2->second << endl;
            }
        } else {
            cout << "Key '" << item1.first << "' is present in the first file but not in the second file" << endl;
        }
    }
    // free all resrouces
}

void ModelXmlParser::run()
{
    printf("libxml version: %s\n", LIBXML_DOTTED_VERSION);

    if (m_iWorkerMode == (int)XmlWokerMode::MODE_CREATE_ITEMS)
    {
        CreateXmlItems();
    }
    else if (m_iWorkerMode == (int)XmlWokerMode::MODE_COMPARE_TWO_FILES)
    {
        CompareTwoFiles();
    }
}
