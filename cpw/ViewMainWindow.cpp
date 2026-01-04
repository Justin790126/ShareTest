#include "ViewMainWindow.h"

ViewMainWindow::ViewMainWindow(QWidget *parent) : QMainWindow(parent) {
    setupUi();
}

ViewMainWindow::~ViewMainWindow() {
    // Qt 的父子物件機制會自動處理子元件釋放，但 scene 需要手動處理或由 view 管理
}

void ViewMainWindow::setupUi() {
    m_splitter = new QSplitter(Qt::Horizontal, this);
    
    treeView = new QTreeView(m_splitter);
    graphicsView = new QGraphicsView(m_splitter);
    scene = new QGraphicsScene(this);
    
    graphicsView->setScene(scene);
    graphicsView->setBackgroundBrush(Qt::black); // 設定 EDA 常用的黑色背景

    m_splitter->addWidget(treeView);
    m_splitter->addWidget(graphicsView);
    
    // 設定左右比例，讓 TreeView 不要佔太多空間
    m_splitter->setStretchFactor(0, 1);
    m_splitter->setStretchFactor(1, 4);

    setCentralWidget(m_splitter);
    setWindowTitle("cpw - CellProfile MVC Viewer");
    resize(1200, 800);
}