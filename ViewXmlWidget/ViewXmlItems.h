
#ifndef VIEW_XML_ITEMS_H
#define VIEW_XML_ITEMS_H

#include <QtGui>

class ViewXmlItems : public QTreeWidgetItem
{
    public:
        ViewXmlItems(QTreeWidget *parent=NULL, const QStringList &strings=QStringList());
};

#endif /* VIEW_XML_ITEMS_H */