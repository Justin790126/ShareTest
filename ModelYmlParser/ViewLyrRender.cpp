#include "ViewYmlDisplay.h"


ViewYmlDisplay::ViewYmlDisplay(QWidget *parent)
    : QWidget(parent)
{
    // Set the widget's size and window title
    setFixedSize(300, 200);
    setWindowTitle("My Custom Widget");


    img1 = QImage(300, 300, QImage::Format_ARGB32);
    img2 = QImage(300, 300, QImage::Format_ARGB32);
    img3 = QImage(300, 300, QImage::Format_ARGB32);

    // Draw shapes on each image
    QPainter painter;
    int cellSize = 100;

    // Image 1: Rectangles
    painter.begin(&img1);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            painter.fillRect(i * cellSize, j * cellSize, cellSize, cellSize, Qt::white);
        }
    }
    painter.end();

    // Image 2: Circles
    painter.begin(&img2);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            // painter.setBrush(Qt::red);
            painter.drawEllipse(i * cellSize + 10, j * cellSize + 10, cellSize - 20, cellSize - 20);
        }
    }
    painter.end();

    // Image 3: Polygons
    painter.begin(&img3);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            QPolygon polygon;
            polygon << QPoint(i * cellSize + 10, j * cellSize + 10)
                    << QPoint(i * cellSize + cellSize - 10, j * cellSize + 10)
                    << QPoint(i * cellSize + cellSize / 2, j * cellSize + cellSize - 10);
            // painter.setBrush(Qt::green);
            painter.drawPolygon(polygon);
        }
    }
    painter.end();
}

void ViewYmlDisplay::paintEvent(QPaintEvent *event)
{
    QPainter painter(this);

    // Draw the blended image
    painter.drawImage(0, 0, img1);
    painter.drawImage(0, 0, img2);
    painter.drawImage(0, 0, img3);
}
