#include "ViewMainWindow.h"
#include <QVBoxLayout>
#include <QSplitter>

ViewMainWindow::ViewMainWindow(QWidget *parent) : QMainWindow(parent) {
    QSplitter *splitter = new QSplitter(Qt::Horizontal, this);
    QWidget *left = new QWidget(splitter);
    QVBoxLayout *vbox = new QVBoxLayout(left);

    searchEdit = new QLineEdit(left);
    searchEdit->setPlaceholderText("無腦搜尋...");
    treeView = new QTreeView(left);
    treeView->setHeaderHidden(true);

    vbox->addWidget(searchEdit);
    vbox->addWidget(treeView);

    graphicsView = new QGraphicsView(splitter);
    scene = new QGraphicsScene(this);
    graphicsView->setScene(scene);
    graphicsView->setBackgroundBrush(Qt::black);

    splitter->setStretchFactor(1, 4);
    setCentralWidget(splitter);
    resize(1200, 800);
}