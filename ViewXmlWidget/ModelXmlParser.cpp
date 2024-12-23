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
void ModelXmlParser::TraverseXmlTree(xmlNode *node, string path, int level, std::map<string, ViewXmlItems *> &map, ViewXmlItems *item)
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
                tagAttr = tag + "/" + string((char *)attr->name) + "/" + val;

                // cout << "Child node column 1: " << attr->name << endl;
                string arrName = string((char *)attr->name);
                attrIt = new ViewXmlItems(tagIt);
                attrIt->setText(1, arrName.c_str());
                attrIt->setText(2, val.c_str());
                attrIt->SetMapKey(tagAttr);
                attrIt->SetAttrValue(val);

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

            if (m_iVerbose)
                cout << "[TAG/ATTR/CONTENT] " << tagAttrContent << " [Content]: " << value << endl;
            if (item)
            {
                item->SetHasContent(qs.length() > 0);
                item->SetContent(value);
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

void ModelXmlParser::CreateXmlItems()
{
    m_xmlDoc = xmlReadFile(m_sFname1.c_str(), NULL, 0);
    if (!m_xmlDoc)
    {
        cerr << "Failed to parse XML file: " << m_sFname1.c_str() << endl;
        return;
    }

    xmlNode *root = xmlDocGetRootElement(m_xmlDoc);
    if (!root)
    {
        cerr << "No root element in XML file: " << m_sFname1.c_str() << endl;
        xmlFreeDoc(m_xmlDoc);
        return;
    }

    // const auto cb = [this](xmlNode *node)
    // {
    //     PrintNodeInfo(node);
    // };
    string path = "";
    int level = 0;
    m_mKeyItems.clear();
    TraverseXmlTree(root, path, level, m_mKeyItems);

    // print map
    // for(auto& it : map) {
    //     std::cout << it.first << ": " << it.second << std::endl;
    // }

    emit AllPageReaded();
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
    cout << "Comparing two XML files: " << m_sFname1.c_str() << " and " << m_sFname2.c_str() << endl;
    xmlDocPtr doc1 = xmlParseFile(m_sFname1.c_str());
    xmlDocPtr doc2 = xmlParseFile(m_sFname2.c_str());
    if (!doc1 || !doc2)
    {
        cerr << "Failed to parse XML files" << endl;
        if (doc1)
            xmlFreeDoc(doc1);
        if (doc2)
            xmlFreeDoc(doc2);
        return;
    }

    xmlNodePtr root1 = xmlDocGetRootElement(doc1);
    xmlNodePtr root2 = xmlDocGetRootElement(doc2);
    if (!root1 || !root2 || xmlStrcmp(root1->name, root2->name) != 0)
    {
        cerr << "Root elements do not match" << endl;
        xmlFreeDoc(doc1);
        xmlFreeDoc(doc2);
        return;
    }
    vector<XmlDiff> diffs;
    CompareNodes(root1, root2, "");

    // print diffs
    for (const auto &diff : diffs)
    {
        cout << diff.path << ": " << diff.node1 << " -> " << diff.node2 << " - " << diff.message << endl;
    }

    xmlFreeDoc(doc1);
    xmlFreeDoc(doc2);
    xmlCleanupParser();
    emit AllPageReaded();
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
