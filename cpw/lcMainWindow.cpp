#include "lcMainWindow.h"
#include <QPen>

lcMainWindow::lcMainWindow() {
    m_view = new ViewMainWindow();
    m_model = new ModelCellProfile();
    m_view->treeView->setModel(m_model);
    
    connect(m_view->treeView, SIGNAL(clicked(QModelIndex)), this, SLOT(onCellClicked(QModelIndex)));
    connect(m_model, SIGNAL(loadingFinished()), this, SLOT(onLoadingFinished()));
    m_view->show();
}

void lcMainWindow::loadProfile(const QString &fileName) {
    m_model->startLoading(fileName);
}

void lcMainWindow::onLoadingFinished() {
    // 展開 Row 0 即 "Top Cells" 分組
    QModelIndex topGroup = m_model->index(0, 0, QModelIndex());
    m_view->treeView->expand(topGroup);
}

void lcMainWindow::onCellClicked(const QModelIndex &index) {
    m_view->scene->clear();
    CellEntry* item = m_model->itemFromIndex(index);
    // 判定點擊的是否為 Cell 而非 Group Header
    if (item && item->parent() != NULL && item->parent()->parent() != NULL) {
        m_view->scene->addRect(item->rect(), QPen(Qt::red, 0), QBrush(QColor(255,0,0,50)));
        m_view->graphicsView->fitInView(item->rect(), Qt::KeepAspectRatio);
    }
}