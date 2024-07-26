#include "webpage.h"
#include <QPainter>
#include <QMouseEvent>

Webpage::Webpage(QWidget *parent) : QWidget(parent), zoomFactor(1.0), drag(false) {}

void Webpage::paintEvent(QPaintEvent *event) {
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);
    painter.scale(zoomFactor, zoomFactor);
    painter.translate(panOffset);
    painter.drawRect(20, 20, 100, 100);
}

void Webpage::mousePressEvent(QMouseEvent *event) {
    if (event->button() == Qt::LeftButton) {
        lastPos = event->pos();
        drag = true;
    }
}

void Webpage::mouseMoveEvent(QMouseEvent *event) {
    if (drag) {
        QPoint delta = event->pos() - lastPos;
        panOffset += delta;
        lastPos = event->pos();
        update();
    }
}

void Webpage::mouseReleaseEvent(QMouseEvent *event) {
    if (event->button() == Qt::LeftButton) {
        drag = false;
    }
}

void Webpage::wheelEvent(QWheelEvent *event) {
    if (event->delta() > 0) {
        zoomFactor *= 1.2;
    } else {
        zoomFactor /= 1.2;
    }
    update();
}

void Webpage::contextMenuEvent(QContextMenuEvent *event)  {
        QMenu contextMenu(tr("Context Menu"), this);

        QAction action1("Action 1", this);
        QAction action2("Action 2", this);
        QAction action3("Action 3", this);

        // Connect actions to slots (for demonstration purposes)
        connect(&action1, SIGNAL(triggered), this, SLOT(onAction1Triggered));
        connect(&action2, SIGNAL(triggered), this, SLOT(onAction1Triggered));
        connect(&action3, SIGNAL(triggered), this, SLOT(onAction1Triggered));

        contextMenu.addAction(&action1);
        contextMenu.addAction(&action2);
        contextMenu.addAction(&action3);

        contextMenu.exec(event->globalPos()); // Show the menu at the cursor position
    }
