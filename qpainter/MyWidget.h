#include <QApplication>
#include <QWidget>
#include <QPainter>
#include <QPixmap>
#include <QPen>
#include <QtGui>
#include <QtCore>
#include <iostream>

using namespace std;

class MyWidget : public QWidget {
    Q_OBJECT
public:
    MyWidget(QWidget *parent = NULL) : QWidget(parent) {
        // Initialize cache
        cache = QPixmap(size());
        cache.fill(Qt::white);
        renderToCache();
    }

private slots:
    void clearCache() {
        cout << "clear" << endl;
        cache.fill(Qt::white);
        update(); // Trigger a repaint
    }

    void render() {
        update();
    }

    void renderPartial() {
        QRect rect(320,0,160,220);
        renderToCacheRect(rect);
    }

    void renderToCache() {
        QPainter painter(&cache);

        // Drawing Text
        painter.drawText(10, 10, "Hello, World!");

        // Drawing Lines
        painter.drawLine(10, 30, 100, 30);

        // Drawing Rectangles
        QRect rect(10, 50, 100, 50);
        painter.drawRect(rect);

        // Drawing Ellipses
        painter.drawEllipse(10, 110, 100, 50);

        // Filling Rectangles
        painter.fillRect(10, 170, 100, 50, Qt::blue);

        // Filling Ellipses
        painter.setBrush(Qt::red);
        painter.drawEllipse(10, 230, 100, 50);

        // Drawing Images
        QPixmap pixmap("image.png");
        painter.drawPixmap(10, 290, pixmap);

        // Setting Pen Properties
        QPen pen(Qt::red, 2, Qt::DashLine);
        painter.setPen(pen);
        painter.drawLine(10, 350, 100, 350);

        // Drawing Polygon
        QPolygon polygon;
        polygon << QPoint(150, 30) << QPoint(200, 80) << QPoint(250, 30) << QPoint(270, 50);
        // polygon << QPoint(0, 0) << QPoint(10, 0) << QPoint(10,10) << QPoint(0, 10);
        painter.drawPolygon(polygon);
    }

    

protected:
    void paintEvent(QPaintEvent *event) override {
        Q_UNUSED(event);
        QPainter painter(this);
        cout << "paint event" << endl;
        // Draw cached pixmap
        painter.drawPixmap(0, 0, cache);
    }

private:
    QPixmap cache;

    void renderToCacheRect(const QRect &rect = QRect()) {
        QPainter painter(&cache);

        // Drawing Text
        painter.drawText(10, 10, "Hello, World!");

        // Drawing Lines
        painter.drawLine(10, 30, 100, 30);

        // Drawing Rectangles
        QRect rectToDraw(320, 50, 100, 50);
        painter.drawRect(rectToDraw);

        // Drawing Ellipses
        painter.drawEllipse(320, 110, 100, 50);

        // Filling Rectangles
        painter.fillRect(320, 170, 100, 50, Qt::blue);

        // Filling Ellipses
        painter.setBrush(Qt::red);
        painter.drawEllipse(10, 230, 100, 50);

        // Drawing Images
        QPixmap pixmap("image.png");
        painter.drawPixmap(10, 290, pixmap);

        // Setting Pen Properties
        QPen pen(Qt::red, 2, Qt::DashLine);
        painter.setPen(pen);
        painter.drawLine(10, 350, 100, 350);

        // Drawing Polygon
        QPolygon polygon;
        polygon << QPoint(150, 30) << QPoint(200, 80) << QPoint(250, 30);
        painter.drawPolygon(polygon);

        // Render only the specified rectangle if provided
        if (!rect.isNull()) {
            painter.setClipRect(rect); // Set the clipping region to the specified rectangle
            painter.setClipping(true);
        }
    }


    
};