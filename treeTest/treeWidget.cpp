#include "treeWidget.h"
#include <QPainter>
#include <QImage>

TreeWidget::TreeWidget(QWidget* parent) : QWidget(parent) {
        setFixedSize(640, 480); // Set the fixed size of the widget

        // Insert 20,000 nodes with random x, y, width, and height
        for (int i = 0; i < 1000; ++i) {
            tree.insert(i);
        }
    }

void TreeWidget::update() {
        std::cout << "In-order traversal: " << std::endl;

        
        img = QImage(width(), height(), QImage::Format_ARGB32);
        QPainter painter(&img);
        tree.setPainter(&painter);

        tree.traverseInOrder(printCallback<int>);
        std::cout << std::endl;
        QWidget::update();
    }

void TreeWidget::paintEvent(QPaintEvent* event) {
 
        QPainter painter(this);
        painter.drawImage(0,0,img);
    }