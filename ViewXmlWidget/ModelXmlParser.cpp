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

    string path1 = "";
    int level1 = 0;
    m_mKeyStatistics1.clear();
    TraverseXmlTree(root1, path1, level1, m_mKeyStatistics1);
    m_mKeyItems1.clear();
    TraverseXmlTree(root1, path1, level1, m_mKeyItems1, NULL, twContainer1);
    // print all keys in m_mKeyItems1
    if (m_iVerbose)
        for (auto &item : m_mKeyItems1)
        {
            cout << item.first << ": " << item.second << endl;
        }

    emit AllPageReaded(twContainer1);
    // iterate m_mKeyStatistics1
    if (m_iVerbose)
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
    TraverseXmlTree(root2, path2, level2, m_mKeyItems2, NULL, twContainer2);

    emit AllPageReaded(twContainer2);
    // iterate m_mKeyStatistics2
    if (m_iVerbose)
        for (auto &item : m_mKeyStatistics2)
        {
            cout << item.first << ": " << item.second << endl;
        }

    // compare m_mKeyStatistics1 and m_mKeyStatistics2, then get the items from m_mKeyItems1 and m_mKeyItems2
    for (auto &item1 : m_mKeyStatistics1) {
        auto it2 = m_mKeyStatistics2.find(item1.first);
        if (it2!= m_mKeyStatistics2.end()) {
            if (it2->second!= item1.second) {
                // cout << "Key '" << item1.first << "' has different counts in both files: " << item1.second << " vs " << it2->second << endl;
            }
        } else {
            string key1 = item1.first;
            if (m_mKeyItems1.count(key1) > 0) {
                ViewXmlItems *item1 = m_mKeyItems1[key1];
                item1->SetHighlighted(true);
            }
            // cout << "Key '" << item1.first << "' is present in the first file but not in the second file" << endl;
        }
    }
    for (auto &item2 : m_mKeyStatistics2) {
        auto it1 = m_mKeyStatistics1.find(item2.first);
        if (it1 != m_mKeyStatistics1.end()) {
            if (it1->second!= item2.second) {
                // cout << "Key '" << item2.first << "' has different counts in both files: " << item2.second << " vs " << it1->second << endl;
            }
        } else {
            string key2 = item2.first;
            if (m_mKeyItems2.count(key2) > 0) {
                ViewXmlItems *item2 = m_mKeyItems2[key2];
                item2->SetHighlighted(true);
            }
            // cout << "Key '" << item2.first << "' is present in the second file but not in the first file" << endl;
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
