#include <QtGui/QApplication>
#include <QtGui/QPainter>
#include <QtGui/QGraphicsScene>
#include <QtGui/QGraphicsView>
#include <QtGui/QGraphicsItem>
#include <QtCore/QThreadPool>
#include <QtCore/QRunnable>
#include <QStyleOption>
#include <vector>

// Structure to store tile data
struct TileData {
    QImage image;
    int x, y;
    TileData() : x(0), y(0) {}
    TileData(const QImage& img, int xPos, int yPos) : image(img), x(xPos), y(yPos) {}
};

// Task to process a single tile (simplified OASIS rendering)
class TileTask : public QRunnable {
public:
    TileTask(std::vector<TileData>& tiles, int index, int tileX, int tileY, int tileWidth, int tileHeight)
        : tiles(tiles), index(index), tileX(tileX), tileY(tileY), tileWidth(tileWidth), tileHeight(tileHeight) {
        setAutoDelete(true);
    }

    void run() {
        QImage tile(tileWidth, tileHeight, QImage::Format_ARGB32);
        tile.fill(qRgba(0, 0, 0, 0));
        QPainter painter(&tile);
        // Simulate OASIS rendering (replace with actual OASIS parsing/drawing)
        for (int y = 0; y < tileHeight; ++y) {
            for (int x = 0; x < tileWidth; ++x) {
                int r = (tileX + x) % 256;
                int g = (tileY + y) % 256;
                int b = ((tileX + x) + (tileY + y)) % 256;
                painter.setPen(QColor(r, g, b));
                painter.drawPoint(x, y);
            }
        }
        tiles[index] = TileData(tile, tileX, tileY);
    }

private:
    std::vector<TileData>& tiles;
    int index, tileX, tileY, tileWidth, tileHeight;
};

// Custom QGraphicsItem for OASIS layout
class OasisGraphicsItem : public QGraphicsItem {
public:
    OasisGraphicsItem(const QString& /*oasisFile*/, QGraphicsItem* parent = 0)
        : QGraphicsItem(parent), isDragging(false), imageWidth(2048), imageHeight(2048), tileSize(256) {
        setFlag(QGraphicsItem::ItemIsSelectable, true); // Enable selection
        // Initialize tiles
        int tilesX = (imageWidth + tileSize - 1) / tileSize;
        int tilesY = (imageHeight + tileSize - 1) / tileSize;
        int totalTiles = tilesX * tilesY;
        tiles.resize(totalTiles);
        QThreadPool threadPool;
        int tileIndex = 0;
        for (int ty = 0; ty < tilesY; ++ty) {
            for (int tx = 0; tx < tilesX; ++tx) {
                int tileX = tx * tileSize;
                int tileY = ty * tileSize;
                int tileWidth = qMin(tileSize, imageWidth - tileX);
                int tileHeight = qMin(tileSize, imageHeight - tileY);
                TileTask* task = new TileTask(tiles, tileIndex, tileX, tileY, tileWidth, tileHeight);
                threadPool.start(task);
                ++tileIndex;
            }
        }
        threadPool.waitForDone();
    }

    QRectF boundingRect() const {
        return QRectF(0, 0, imageWidth, imageHeight);
    }

    void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* /*widget*/) {
        painter->setClipRect(option->exposedRect);
        for (std::vector<TileData>::const_iterator it = tiles.begin(); it != tiles.end(); ++it) {
            if (!it->image.isNull()) {
                QRectF tileRect(it->x, it->y, it->image.width(), it->image.height());
                if (tileRect.intersects(option->exposedRect)) {
                    painter->drawImage(it->x, it->y, it->image);
                }
            }
        }
    }

    // Simulate resizeEvent by updating size (e.g., toggle between sizes)
    void resize(int newWidth, int newHeight) {
        prepareGeometryChange(); // Notify scene of geometry change
        imageWidth = newWidth;
        imageHeight = newHeight;
        // Recompute tiles (simplified; in practice, reload OASIS data)
        tiles.clear();
        int tilesX = (imageWidth + tileSize - 1) / tileSize;
        int tilesY = (imageHeight + tileSize - 1) / tileSize;
        int totalTiles = tilesX * tilesY;
        tiles.resize(totalTiles);
        QThreadPool threadPool;
        int tileIndex = 0;
        for (int ty = 0; ty < tilesY; ++ty) {
            for (int tx = 0; tx < tilesX; ++tx) {
                int tileX = tx * tileSize;
                int tileY = ty * tileSize;
                int tileWidth = qMin(tileSize, imageWidth - tileX);
                int tileHeight = qMin(tileSize, imageHeight - tileY);
                TileTask* task = new TileTask(tiles, tileIndex, tileX, tileY, tileWidth, tileHeight);
                threadPool.start(task);
                ++tileIndex;
            }
        }
        threadPool.waitForDone();
        update(); // Trigger repaint
    }

protected:
    void mousePressEvent(QGraphicsSceneMouseEvent* event) {
        if (event->button() == Qt::LeftButton) {
            isDragging = true;
            lastPos = event->scenePos();
            // Simulate resize on right-click for testing
        } else if (event->button() == Qt::RightButton) {
            // Toggle between two sizes to simulate resizeEvent
            if (imageWidth == 2048) {
                resize(4096, 4096); // Double size
            } else {
                resize(2048, 2048); // Restore original size
            }
        }
        QGraphicsItem::mousePressEvent(event);
    }

    void mouseMoveEvent(QGraphicsSceneMouseEvent* event) {
        if (isDragging) {
            // Pan by moving the item
            QPointF delta = event->scenePos() - lastPos;
            setPos(pos() + delta);
            lastPos = event->scenePos();
        }
        QGraphicsItem::mouseMoveEvent(event);
    }

    void mouseReleaseEvent(QGraphicsSceneMouseEvent* event) {
        if (event->button() == Qt::LeftButton) {
            isDragging = false;
        }
        QGraphicsItem::mouseReleaseEvent(event);
    }

private:
    std::vector<TileData> tiles;
    int imageWidth, imageHeight, tileSize;
    bool isDragging;
    QPointF lastPos;
};

int main(int argc, char* argv[]) {
    QApplication app(argc, argv);

    QGraphicsScene scene;
    scene.setSceneRect(0, 0, 400, 300);

    OasisGraphicsItem* oasisItem = new OasisGraphicsItem("layout.oas");
    scene.addItem(oasisItem);
    oasisItem->setPos(0, 0);

    QGraphicsView view(&scene);
    view.setWindowTitle("OASIS Layout with Events in QGraphics Framework");
    view.resize(450, 350);
    view.setRenderHint(QPainter::Antialiasing);
    view.setDragMode(QGraphicsView::ScrollHandDrag);
    view.setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
    view.setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOn);

    view.show();

    return app.exec();
}