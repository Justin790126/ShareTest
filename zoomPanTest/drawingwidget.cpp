#include "drawingwidget.h"
#include <QPainter>
#include <QMouseEvent>

DrawingWidget::DrawingWidget(QWidget *parent) : QWidget(parent), zoomFactor(1.0), drag(false) {}

void DrawingWidget::paintEvent(QPaintEvent *event) {
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);
    painter.scale(zoomFactor, zoomFactor);
    painter.translate(panOffset);
    painter.drawRect(20, 20, 100, 100);
}

void DrawingWidget::mousePressEvent(QMouseEvent *event) {
    if (event->button() == Qt::LeftButton) {
        lastPos = event->pos();
        drag = true;
    }
}

void DrawingWidget::mouseMoveEvent(QMouseEvent *event) {
    if (drag) {
        QPoint delta = event->pos() - lastPos;
        panOffset += delta;
        lastPos = event->pos();
        update();
    }
}

void DrawingWidget::mouseReleaseEvent(QMouseEvent *event) {
    if (event->button() == Qt::LeftButton) {
        drag = false;
    }
}

void DrawingWidget::wheelEvent(QWheelEvent *event) {
    if (event->delta() > 0) {
        zoomFactor *= 1.2;
    } else {
        zoomFactor /= 1.2;
    }
    update();
}
