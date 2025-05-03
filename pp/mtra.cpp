#include <QtGui/QApplication>
#include <QtGui/QImage>
#include <QtGui/QPainter>
#include <QtCore/QRunnable>
#include <QtCore/QThreadPool>
#include <QtCore/QMutex>
#include <QtCore/QTime>
#include <QtCore/QDebug>
#include <vector>

class TileTask : public QRunnable {
public:
    TileTask(QImage* outputImage, int tileX, int tileY, int tileWidth, int tileHeight, QMutex* mutex)
        : outputImage(outputImage), tileX(tileX), tileY(tileY), tileWidth(tileWidth), tileHeight(tileHeight), mutex(mutex) {
        setAutoDelete(true);
    }

    void run() {
        QImage tile(tileWidth, tileHeight, QImage::Format_ARGB32);
        tile.fill(qRgba(0, 0, 0, 0));

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

        QMutexLocker locker(mutex);
        QPainter outputPainter(outputImage);
        outputPainter.drawImage(tileX, tileY, tile);
    }

private:
    QImage* outputImage;
    int tileX, tileY, tileWidth, tileHeight;
    QMutex* mutex;
};

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);

    const int imageWidth = 4096;
    const int imageHeight = 4096;
    const int tileSize = 2048;

    QImage outputImage(imageWidth, imageHeight, QImage::Format_ARGB32);
    outputImage.fill(qRgba(0, 0, 0, 0));

    // Start timing
    QTime timer;
    timer.start();

    QThreadPool threadPool;
    QMutex mutex;

    int tilesX = (imageWidth + tileSize - 1) / tileSize;
    int tilesY = (imageHeight + tileSize - 1) / tileSize;

    for (int ty = 0; ty < tilesY; ++ty) {
        for (int tx = 0; tx < tilesX; ++tx) {
            int tileX = tx * tileSize;
            int tileY = ty * tileSize;
            int tileWidth = qMin(tileSize, imageWidth - tileX);
            int tileHeight = qMin(tileSize, imageHeight - tileY);

            TileTask* task = new TileTask(&outputImage, tileX, tileY, tileWidth, tileHeight, &mutex);
            threadPool.start(task);
        }
    }

    threadPool.waitForDone();

    // Measure elapsed time
    int elapsed = timer.elapsed();
    qDebug() << "Parallel processing time:" << elapsed << "ms";

    outputImage.save("output_parallel.png");

    qDebug() << "Image processing complete. Saved as output_parallel.png";

    return 0;
}