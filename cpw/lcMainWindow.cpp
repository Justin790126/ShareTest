#include "lcMainWindow.h"
#include <QPen>
#include <QBrush>
#include <QColor>

lcMainWindow::lcMainWindow() {
    m_view = new ViewMainWindow();
    m_model = new ModelCellProfile();
    
    // 連結 Model 與 View
    m_view->treeView->setModel(m_model);

    connect(m_view->treeView, SIGNAL(clicked(QModelIndex)), 
            this, SLOT(onCellClicked(QModelIndex)));

    m_view->show();
}

lcMainWindow::~lcMainWindow() {
    delete m_model;
    delete m_view;
}

void lcMainWindow::generateAndLoad(int count) {
    m_model->loadFakeData(count);
    m_view->treeView->expandAll();
}

void lcMainWindow::onCellClicked(const QModelIndex &index) {
    if (!index.isValid()) return;

    m_view->scene->clear();
    CellEntry* item = m_model->itemFromIndex(index);
    
    if (item) {
        // 畫出紅色外框的 Cell BBox 
        m_view->scene->addRect(item->rect(), 
                               QPen(Qt::red, 0), // 0 表示細線，不隨縮放變粗
                               QBrush(QColor(255, 0, 0, 50)));

        m_view->graphicsView->fitInView(item->rect(), Qt::KeepAspectRatio);
    }
}