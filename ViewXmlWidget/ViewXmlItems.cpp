
#include "ViewXmlItems.h"

ViewXmlItems::ViewXmlItems(QTreeWidget *parent) : QTreeWidgetItem(parent)
{
}

ViewXmlItems::ViewXmlItems(QTreeWidgetItem *parent) : QTreeWidgetItem(parent)
{
    
}

void ViewXmlItems::SetHighlighted(bool highlighted)
{
    // apply to all columns
    for (int i = 0; i < columnCount(); i++) {
        setBackgroundColor(i, highlighted? QColor(255, 0, 0) : QColor(Qt::white));
        setTextColor(i, highlighted? QColor(255, 255, 255) : QColor(Qt::black));
    }
}

// Add more functions and properties as needed