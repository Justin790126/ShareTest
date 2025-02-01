#include "ra.h"

Ra::Ra(QWidget *parent) : QWidget(parent) {
    
    setWindowTitle("Qt Render Area");

    resize(1024,768);
}

void Ra::paintEvent(QPaintEvent* event)
{
    QPainter painter(this);
    if (m_pImg) {
        cout << "paint event: " << m_pImg << endl;
        painter.drawImage(0, 0, *m_pImg);
    }
}

void Ra::keyPressEvent(QKeyEvent* event)
{
    switch (event->key()) {
        case Qt::Key_Q:
            close();
            break;
        default:
        // other key events
            update();
            break;
    }
}