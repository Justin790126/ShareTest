#include <QtGui/QApplication>
#include <QtGui/QImage>
#include <QtGui/QPainter>
#include <QtCore/QRunnable>
#include <QtCore/QThreadPool>
#include <QtCore/QTime>
#include <QtCore/QDebug>
#include <vector>

// Structure to store tile data
struct TileData {
    QImage image;
    int x, y;
    TileData() : x(0), y(0) {} // Default constructor for vector initialization
    TileData(const QImage& img, int xPos, int yPos) : image(img), x(xPos), y(yPos) {}
};

// Task to process a single tile
class TileTask : public QRunnable {
public:
    TileTask(std::vector<TileData>& tiles, int index, int tileX, int tileY, int tileWidth, int tileHeight)
        : tiles(tiles), index(index), tileX(tileX), tileY(tileY), tileWidth(tileWidth), tileHeight(tileHeight) {
        setAutoDelete(true);
    }

    void run() {
        // Create a tile image
        QImage tile(tileWidth, tileHeight, QImage::Format_ARGB32);
        tile.fill(qRgba(0, 0, 0, 0));

        // Draw on the tile (gradient pattern)
        QPainter painter(&tile);
        for (int y = 0; y < tileHeight; ++y) {
            for (int x = 0; x < tileWidth; ++x) {
                int r = (tileX + x) % 256;
                int g = (tileY + y) % 256;
                int b = ((tileX + x) + (tileY + y)) % 256;
                painter.setPen(QColor(r, g, b));
                painter.drawPoint(x, y);
            }
        }

        // Store the tile at the pre-assigned index
        tiles[index] = TileData(tile, tileX, tileY);
    }

private:
    std::vector<TileData>& tiles;
    int index;
    int tileX, tileY, tileWidth, tileHeight;
};

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);

    // Define the large image dimensions
    const int imageWidth = 2048;
    const int imageHeight = 2048;
    const int tileSize = 256;

    // Create the output image
    QImage outputImage(imageWidth, imageHeight, QImage::Format_ARGB32);
    outputImage.fill(qRgba(0, 0, 0, 0));

    // Start timing
    QTime timer;
    timer.start();

    // Calculate number of tiles
    int tilesX = (imageWidth + tileSize - 1) / tileSize;
    int tilesY = (imageHeight + tileSize - 1) / tileSize;
    int totalTiles = tilesX * tilesY;

    // Pre-allocate the vector for tiles
    std::vector<TileData> tiles(totalTiles);

    // Thread pool
    QThreadPool threadPool;

    // Create and start tasks for each tile
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

    // Wait for all tasks to complete
    threadPool.waitForDone();

    // Merge all tiles into the output image (single-threaded)
    QPainter outputPainter(&outputImage);
    for (std::vector<TileData>::const_iterator it = tiles.begin(); it != tiles.end(); ++it) {
        if (!it->image.isNull()) { // Ensure the tile was populated
            outputPainter.drawImage(it->x, it->y, it->image);
        }
    }

    // Measure elapsed time
    int elapsed = timer.elapsed();
    qDebug() << "Parallel processing time (indexed, no mutex):" << elapsed << "ms";

    // Save the final image
    outputImage.save("output_parallel_indexed.png");

    qDebug() << "Image processing complete. Saved as output_parallel_indexed.png";

    return 0;
}