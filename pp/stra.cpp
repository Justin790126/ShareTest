#include <QtGui/QApplication>
#include <QtGui/QImage>
#include <QtGui/QPainter>
#include <QtCore/QTime>
#include <QtCore/QDebug>

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

    // Process each tile sequentially
    for (int ty = 0; ty < tilesY; ++ty) {
        for (int tx = 0; tx < tilesX; ++tx) {
            int tileX = tx * tileSize;
            int tileY = ty * tileSize;
            int tileWidth = qMin(tileSize, imageWidth - tileX);
            int tileHeight = qMin(tileSize, imageHeight - tileY);

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

            // Merge tile into the output image
            QPainter outputPainter(&outputImage);
            outputPainter.drawImage(tileX, tileY, tile);
        }
    }

    // Measure elapsed time
    int elapsed = timer.elapsed();
    qDebug() << "Single-threaded processing time:" << elapsed << "ms";

    // Save the final image
    outputImage.save("output_single.png");

    qDebug() << "Image processing complete. Saved as output_single.png";

    return 0;
}