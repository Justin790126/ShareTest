#include "ViewProfTreeItem.h"

ViewProfTreeItem::ViewProfTreeItem(const ProfNodeStd::Ptr& node)
    : QTreeWidgetItem(ViewProfTreeItem::Type)
    , m_node(node)
{
    // 依你的 node 欄位填入各欄
    setText(COL_NAME, qs(node->name()));
    setText(COL_TOTAL, qs(node->totalTimeMs()));
    setText(COL_CPU, qs(node->cpuTimeMs()));
    setText(COL_CALLS, qs(node->numCalls()));
    setText(COL_PERCENT, qs(node->percentage()));
    setText(COL_CHILD_PERCENT, qs(node->childProf()));
    setText(COL_MEM, qs(node->memoryMb()));
}

QString ViewProfTreeItem::qs(const std::string& s)
{
    return QString::fromLocal8Bit(s.c_str());
}
