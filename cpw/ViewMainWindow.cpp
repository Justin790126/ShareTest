#include "ViewMainWindow.h"
#include <QVBoxLayout>
#include <QWidget>

ViewMainWindow::ViewMainWindow(QWidget *parent) : QMainWindow(parent) {
    m_splitter = new QSplitter(Qt::Horizontal, this);
    
    QWidget *leftContainer = new QWidget(m_splitter);
    QVBoxLayout *leftLayout = new QVBoxLayout(leftContainer);
    
    searchEdit = new QLineEdit(leftContainer);
    searchEdit->setPlaceholderText("無腦搜尋所有 Cell...");
    
    treeView = new QTreeView(leftContainer);
    treeView->setUniformRowHeights(true);
    treeView->setHeaderHidden(true); // 隱藏表頭，更清爽
    
    leftLayout->addWidget(searchEdit);
    leftLayout->addWidget(treeView);
    
    graphicsView = new QGraphicsView(m_splitter);
    scene = new QGraphicsScene(this);
    graphicsView->setScene(scene);
    graphicsView->setBackgroundBrush(Qt::black);

    m_splitter->addWidget(leftContainer);
    m_splitter->addWidget(graphicsView);
    m_splitter->setStretchFactor(1, 4);

    setCentralWidget(m_splitter);
    resize(1200, 800);
}