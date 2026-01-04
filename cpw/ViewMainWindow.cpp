#include "ViewMainWindow.h"

ViewMainWindow::ViewMainWindow(QWidget *parent) : QMainWindow(parent) {
    QSplitter *split = new QSplitter(Qt::Horizontal, this);
    treeView = new QTreeView(split);
    graphicsView = new QGraphicsView(split);
    scene = new QGraphicsScene(this);
    graphicsView->setScene(scene);
    graphicsView->setBackgroundBrush(Qt::black);
    setCentralWidget(split);
    setWindowTitle("cpw - Threaded MVC Viewer");
    resize(1000, 600);
}