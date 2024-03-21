#include <QtGui>
#include <QtCore/qmath.h>


float qDegreesToRadians(double degree)
{
    return degree * M_PI / 180.0;
}

class CustomGraphicsScene : public QGraphicsScene {
public:
    CustomGraphicsScene(QObject* parent = NULL) : QGraphicsScene(parent) {}

    void drawBackground(QPainter *painter, const QRectF &rect) override {
        // painter->setBrush(Qt::white);
        // painter->fillRect(rect);

        // Draw a polygon in the background
        QPolygonF polygon;
        polygon << QPointF(50, 50) << QPointF(150, 50) << QPointF(150, 150) << QPointF(50, 150);
        painter->setBrush(Qt::green);
        painter->drawPolygon(polygon);

        // Call the base class implementation to draw the default background
        QGraphicsScene::drawBackground(painter, rect);
    }
};

class HexagonItem : public QGraphicsPolygonItem
{
public:
    HexagonItem(QGraphicsItem* parent = NULL)
        : QGraphicsPolygonItem(QPolygonF(), parent), selectedVertexIndex(-1), vertexSize(5)
    {
        setAcceptHoverEvents(true);
        setFlag(QGraphicsItem::ItemIsSelectable);
        setFlag(QGraphicsItem::ItemIsMovable);

        QPointF center(0, 0);
        qreal size = 50; // Adjust the size of the hexagon as needed

        // Calculate vertices of a regular hexagon
        for (int i = 0; i < 6; ++i) {
            qreal angle = 60.0 * i + 30.0;
            QPointF vertex = QPointF(center.x() + size * qCos(qDegreesToRadians(angle)),
                                      center.y() + size * qSin(qDegreesToRadians(angle)));
            hexagon << vertex;
        }

        setPolygon(hexagon);
    }

protected:
    void mousePressEvent(QGraphicsSceneMouseEvent* event) override
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

    void mouseMoveEvent(QGraphicsSceneMouseEvent* event) override
    {
        if (selectedVertexIndex != -1) {
            hexagon[selectedVertexIndex] = event->pos();
            setPolygon(hexagon);
        } else {
            QGraphicsPolygonItem::mouseMoveEvent(event);
        }
    }

    void mouseReleaseEvent(QGraphicsSceneMouseEvent* event) override
    {
        selectedVertexIndex = -1;
        QGraphicsPolygonItem::mouseReleaseEvent(event);
    }

    void hoverMoveEvent(QGraphicsSceneHoverEvent* event) override
    {
        QPointF pos = event->pos();
        for (int i = 0; i < hexagon.size(); ++i) {
            if (QRectF(hexagon[i].x() - vertexSize / 2, hexagon[i].y() - vertexSize / 2,
                        vertexSize, vertexSize).contains(pos)) {
                setCursor(Qt::CrossCursor);
                return;
            }
        }
        setCursor(Qt::ArrowCursor);
        QGraphicsPolygonItem::hoverMoveEvent(event);
    }

private:
    int selectedVertexIndex;
    qreal vertexSize;
    QPolygonF hexagon;
};

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    CustomGraphicsScene scene;
    QGraphicsView view(&scene);

    HexagonItem* hexagonItem = new HexagonItem();
    scene.addItem(hexagonItem);

    view.show();

    return app.exec();
}
