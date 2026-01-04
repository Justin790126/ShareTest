#include "lcMainWindow.h"
#include <QPen>

lcMainWindow::lcMainWindow() {
    m_view = new ViewMainWindow();
    m_model = new ModelCellProfile();
    
    m_proxy = new RecursiveFilterProxy();
    m_proxy->setSourceModel(m_model);
    m_proxy->setDynamicSortFilter(true);

    m_view->treeView->setModel(m_proxy);
    
    connect(m_view->searchEdit, SIGNAL(textChanged(QString)), this, SLOT(onSearchTextChanged(QString)));
    connect(m_view->treeView, SIGNAL(clicked(QModelIndex)), this, SLOT(onCellClicked(QModelIndex)));
    connect(m_model, SIGNAL(loadingFinished()), this, SLOT(onLoadingFinished()));
    
    m_view->show();
}

void lcMainWindow::onSearchTextChanged(const QString &text) {
    m_proxy->setFilterFixedString(text);
    if (!text.isEmpty()) {
        m_view->treeView->expandAll(); // 搜尋時全部展開
    } else {
        onLoadingFinished(); // 清空時恢復預設展開
    }
}

void lcMainWindow::onCellClicked(const QModelIndex &proxyIdx) {
    m_view->scene->clear();
    QModelIndex sourceIdx = m_proxy->mapToSource(proxyIdx);
    CellEntry* item = m_model->itemFromIndex(sourceIdx);
    
    // 只有真正的 Cell (非分類標題) 才畫
    if (item && item->parent() != NULL && item->parent()->parent() != NULL) {
        m_view->scene->addRect(item->rect(), QPen(Qt::red, 0), QBrush(QColor(255,0,0,50)));
        m_view->graphicsView->fitInView(item->rect(), Qt::KeepAspectRatio);
    }
}

void lcMainWindow::loadProfile(const QString &fileName) {
    m_model->startLoading(fileName);
}

void lcMainWindow::onLoadingFinished() {
    // 預設展開 Top Cells
    m_view->treeView->collapseAll();
    m_view->treeView->expand(m_proxy->index(0, 0, QModelIndex()));
}