#pragma once
#include <QtGui/QWidget>
#include <QtCore/QPointF>
#include "layout.h"
#include "selector.h"

static inline long roundToLong(double v)
{
    return (v >= 0.0) ? (long)(v + 0.5) : (long)(v - 0.5);
}

class DemoView : public QWidget {
    Q_OBJECT
public:
    DemoView(QWidget* parent = 0);

protected:
    void paintEvent(QPaintEvent* e);
    void mouseMoveEvent(QMouseEvent* e);

private:
    // world <-> screen mapping
    double umPerPixel; // 1 px = umPerPixel
    QPointF originPx;  // screen origin

    QPointF screenToUm(const QPointF& p) const;
    QPointF umToScreen(const QPointF& p) const;

    FakeLayout m_layout;
    lcSelector m_selector;

    // last query
    QPointF m_usrUm;
    QPointF m_snapUm;
    Edge    m_edge;
    Vertex  m_vertex;
    int     m_res;
};
