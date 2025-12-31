#pragma once

#include <QTreeWidgetItem>
#include <QString>
#include <memory>

#include "ModelProfTree.h" // for ProfNodeStd

class ViewProfTreeItem : public QTreeWidgetItem
{
public:
    // columns index
    enum Col {
        COL_NAME = 0,
        COL_TOTAL,
        COL_CPU,
        COL_CALLS,
        COL_PERCENT,
        COL_CHILD_PERCENT,
        COL_MEM,
        COL_COUNT
    };

    explicit ViewProfTreeItem(const ProfNodeStd::Ptr& node);
    virtual ~ViewProfTreeItem() {}

    std::shared_ptr<const ProfNodeStd> node() const { return m_node; }

private:
    static QString qs(const std::string& s);

private:
    std::shared_ptr<const ProfNodeStd> m_node;
};
