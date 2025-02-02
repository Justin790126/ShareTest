#include <QtGui>
#include <iostream>
#include <chrono>
#include "ra.h"
#include "ModelTreeGen.h"
#include <omp.h>
using namespace std;

QImage blendExample() {
    QImage QImage1(1024, 768, QImage::Format_ARGB32);
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
    ModelTreeGen model;
    model.CreateExampleNode();
    Node* root = model.GetRootNode();

    int alg = atoi(argv[1]);
    QImage img;
    auto start = std::chrono::high_resolution_clock::now();

    if (alg == 0) {
        img = blendExample();
        
    } else if (alg == 1) {
        // calibration
        img = QImage(1024, 768, QImage::Format_ARGB32);
        img.fill(qRgb(255, 255, 255));
        model.SetTargetLyr(0);
        model.SetImage(&img);
        model.draw();

    } else if (alg == 2) {
        // total draw
        img = QImage(1024, 768, QImage::Format_ARGB32);
        model.SetTargetLyr(-1);
        model.SetImage(&img);
        model.draw();

    } else if (alg == 3) {
        // multi-draw and merge
        QImage QImage1(1024, 768, QImage::Format_ARGB32);
        QImage1.fill(qRgb(255, 255, 255));

        vector<QImage> imgs(129, QImage(1024, 768, QImage::Format_ARGB32));
        #pragma omp parallel
        {
            ModelTreeGen m;
            m.SetRootNode(root);
            #pragma omp for nowait 
            {
                
                for (int i = 0; i < 128; i++) {
                    m.SetTargetLyr(i);
                    m.SetImage(&imgs[i]);
                    m.draw();

                    QPainter painter(&QImage1);
                    painter.setCompositionMode(QPainter::CompositionMode_SourceOver);
                    painter.drawImage(0,0, imgs[i]);
                    
                }
            }
        }

        img = std::move(QImage1);
    }
     auto end = std::chrono::high_resolution_clock::now();

    // Compute duration in seconds
    std::chrono::duration<double> elapsed = end - start;
    
    // Print elapsed time in seconds
    std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;

    ra.SetImage(&img);
    ra.update();
    ra.show();
    ra.resize(1024,768);


    return a.exec();
}
