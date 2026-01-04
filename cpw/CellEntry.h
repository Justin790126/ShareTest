#ifndef CELLENTRY_H
#define CELLENTRY_H
#include <QString>
#include <QRectF>
#include <QVector>

class CellEntry {
public:
    CellEntry() : m_offset(0), m_parent(NULL) {}
    CellEntry(QString name, QRectF rect, unsigned long long offset, CellEntry* parent = NULL)
        : m_name(name), m_rect(rect), m_offset(offset), m_parent(parent) {}

    QString name() const { return m_name; }
    QRectF rect() const { return m_rect; }
    CellEntry* parent() { return m_parent; }
    QVector<CellEntry*>& children() { return m_children; }
    void addChild(CellEntry* child) { m_children.append(child); }

    int row() const {
        if (m_parent) return m_parent->m_children.indexOf(const_cast<CellEntry*>(this));
        return 0;
    }

private:
    QString m_name;
    QRectF m_rect;
    unsigned long long m_offset;
    CellEntry* m_parent;
    QVector<CellEntry*> m_children;
};
#endif