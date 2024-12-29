
#ifndef VIEW_XML_ITEMS_H
#define VIEW_XML_ITEMS_H

#include <iostream>
#include <QtGui>

using namespace std;


class ViewXmlItems : public QTreeWidgetItem
{
    public:
        ViewXmlItems(QTreeWidget *parent=NULL);
        ViewXmlItems(QTreeWidgetItem *parent=NULL);
        void SetMapKey(const string &key) { m_sMapKey = key; }
        string GetMapKey() const { return m_sMapKey; }
        void SetAttrValue(const string &value) { m_sAttrValue = value; }
        string GetAttrValue() const { return m_sAttrValue; }
        void SetContent(const string &content) { m_sContent = content; }
        string GetContent() const { return m_sContent; }
        void SetHasContent(bool hasContent) { m_bHasContent = hasContent; }
        bool HasContent() const { return m_bHasContent; }
        void SetHighlighted(bool highlighted);
    private:
        string m_sMapKey;
        string m_sAttrValue;
        string m_sContent;
        bool m_bHasContent = false;
        bool m_bHighlighted = false;
        
};

#endif /* VIEW_XML_ITEMS_H */