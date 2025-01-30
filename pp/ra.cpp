#include "ra.h"

Ra::Ra(QWidget *parent) : QWidget(parent) {
    
    setWindowTitle("Qt Render Area");
}

void Ra::paintEvent(QPaintEvent* event)
{
    QPainter painter(this);
    if (m_pImg) {
        cout << "paint event: " << m_pImg << endl;
        painter.drawImage(0, 0, *m_pImg);
    }
}