#ifndef CELLENTRY_H
#define CELLENTRY_H

#include <QString>
#include <QRectF>
#include <QVector>

class CellEntry {
public:
    CellEntry(QString name, QRectF rect, CellEntry* parent = NULL)
        : m_name(name), m_rect(rect), m_parent(parent) {}
    
    ~CellEntry() { qDeleteAll(m_children); }

    QString name() const { return m_name; }
    QRectF rect() const { return m_rect; }
    CellEntry* parent() const { return m_parent; }
    QVector<CellEntry*>& children() { return m_children; }
    void addChild(CellEntry* child) { m_children.append(child); }

    int row() const {
        if (m_parent) return m_parent->m_children.indexOf(const_cast<CellEntry*>(this));
        return 0;
    }

private:
    QString m_name;
    QRectF m_rect;
    CellEntry* m_parent;
    QVector<CellEntry*> m_children;
};

#endif