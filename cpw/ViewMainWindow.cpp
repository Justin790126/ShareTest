#include "ViewMainWindow.h"

ViewMainWindow::ViewMainWindow(QWidget *parent) : QMainWindow(parent) {
    QSplitter *s = new QSplitter(Qt::Horizontal, this);
    treeView = new QTreeView(s);
    graphicsView = new QGraphicsView(s);
    scene = new QGraphicsScene(this);
    graphicsView->setScene(scene);
    graphicsView->setBackgroundBrush(Qt::black);
    s->setStretchFactor(1, 4);
    setCentralWidget(s);
    resize(1100, 700);
}