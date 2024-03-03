#include "hexagonitem.h"
#include <QRectF>
#include <QtMath>

HexagonItem::HexagonItem(const QList<QPointF>& vertices, QGraphicsItem* parent)
    : QGraphicsPolygonItem(parent), selectedVertexIndex(-1), vertexSize(10), hexagon(vertices)
{
    setAcceptHoverEvents(true);
    setFlag(QGraphicsItem::ItemIsSelectable);
    setFlag(QGraphicsItem::ItemIsMovable);
    updatePolygon();
}

void HexagonItem::mousePressEvent(QGraphicsSceneMouseEvent* event)
{
    QPointF pos = event->pos();
    for (int i = 0; i < hexagon.size(); ++i) {
        if (QRectF(hexagon[i].x() - vertexSize / 2, hexagon[i].y() - vertexSize / 2,
                    vertexSize, vertexSize).contains(pos)) {
            selectedVertexIndex = i;
            return;
        }
    }
    selectedVertexIndex = -1;
    QGraphicsPolygonItem::mousePressEvent(event);
}

void HexagonItem::mouseMoveEvent(QGraphicsSceneMouseEvent* event)
{
    if (selectedVertexIndex != -1) {
        hexagon[selectedVertexIndex] = event->pos();
        updatePolygon();
    } else {
        QGraphicsPolygonItem::mouseMoveEvent(event);
    }
}

void HexagonItem::mouseReleaseEvent(QGraphicsSceneMouseEvent* event)
{
    selectedVertexIndex = -1;
    QGraphicsPolygonItem::mouseReleaseEvent(event);
}

void HexagonItem::hoverMoveEvent(QGraphicsSceneHoverEvent* event)
{
    QPointF pos = event->pos();
    bool overVertex = false;
    for (int i = 0; i < hexagon.size(); ++i) {
        if (QRectF(hexagon[i].x() - vertexSize / 2, hexagon[i].y() - vertexSize / 2,
                    vertexSize, vertexSize).contains(pos)) {
            overVertex = true;
            break;
        }
    }
    setCursor(overVertex ? Qt::CrossCursor : Qt::ArrowCursor);
    QGraphicsPolygonItem::hoverMoveEvent(event);
}

void HexagonItem::updatePolygon()
{
    QPolygonF polygon;
    for (const QPointF& point : hexagon) {
        polygon << point;
    }
    setPolygon(polygon);
}
