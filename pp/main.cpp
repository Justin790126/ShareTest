#include <QtGui>
#include <iostream>
#include "ra.h"
#include "ModelTreeGen.h"

using namespace std;

QImage blendExample() {
    QImage QImage1(640, 480, QImage::Format_ARGB32);
    QImage1.fill(qRgb(255, 255, 255));

    // Draw a rectangle on QImage1
    QPainter painter1(&QImage1);
    painter1.setBrush(QColor(0, 0, 255, 255)); // Solid Blue
    painter1.setPen(Qt::black);
    painter1.drawRect(100, 100, 200, 150); // (x, y, width, height)
    painter1.end();

    QImage QImage2(640, 480, QImage::Format_ARGB32);
    QImage2.fill(Qt::transparent); // Transparent background

    // Draw a polygon on QImage2
    QPainter painter2(&QImage2);
    painter2.setBrush(QColor(255, 0, 0, 128)); // Semi-transparent Red
    painter2.setPen(Qt::black);

    QPolygon polygon;
    polygon << QPoint(200, 200) << QPoint(350, 100)
            << QPoint(500, 200) << QPoint(400, 350) << QPoint(250, 350);

    painter2.drawPolygon(polygon);
    painter2.end();
    
    // draw random polygons to ra
    QPainter painter3(&QImage1);
    painter3.setCompositionMode(QPainter::CompositionMode_SourceOver);
    painter3.drawImage(0,0, QImage2);

    return QImage1;
}

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    Ra ra;
    ModelTreeGen modelTreeGen;
    modelTreeGen.CreateExampleNode();

    QImage img = blendExample();

    ra.SetImage(&img);
    ra.update();
    ra.show();
    ra.resize(1024,768);


    return a.exec();
}
