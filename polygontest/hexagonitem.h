#ifndef HEXAGONITEM_H
#define HEXAGONITEM_H

#include <QMainWindow>
#include <QGraphicsView>
#include <QGraphicsPolygonItem>
#include <QGraphicsSceneMouseEvent>
#include <QGraphicsSceneHoverEvent>

class HexagonItem : public QGraphicsPolygonItem
{
public:
    HexagonItem(const QList<QPointF>& vertices, QGraphicsItem* parent = nullptr);

protected:
    void mousePressEvent(QGraphicsSceneMouseEvent* event) override;
    void mouseMoveEvent(QGraphicsSceneMouseEvent* event) override;
    void mouseReleaseEvent(QGraphicsSceneMouseEvent* event) override;
    void hoverMoveEvent(QGraphicsSceneHoverEvent* event) override;

private:
    void updatePolygon();

    int selectedVertexIndex;
    qreal vertexSize;
    QList<QPointF> hexagon;
};

#endif // HEXAGONITEM_H
