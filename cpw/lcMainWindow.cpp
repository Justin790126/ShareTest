#include "lcMainWindow.h"
#include <QPen>

lcMainWindow::lcMainWindow(QObject *parent) : QObject(parent) {
    m_view = new ViewMainWindow();
    m_model = new ModelCellProfile();
    m_proxy = new RecursiveProxy();
    m_proxy->setSourceModel(m_model);
    m_proxy->setDynamicSortFilter(true);
    m_view->treeView->setModel(m_proxy);

    connect(m_view->searchEdit, SIGNAL(textChanged(QString)), this, SLOT(onSearch(QString)));
    connect(m_view->treeView, SIGNAL(clicked(QModelIndex)), this, SLOT(onClick(QModelIndex)));
    connect(m_model, SIGNAL(loadingFinished()), this, SLOT(onDone()));
    m_view->show();
}

void lcMainWindow::loadProfile(const QString &f) { m_model->startLoading(f); }
void lcMainWindow::onSearch(const QString &t) { 
    m_proxy->setFilterFixedString(t); 
    if(!t.isEmpty()) m_view->treeView->expandAll(); 
}
void lcMainWindow::onClick(const QModelIndex &idx) {
    m_view->scene->clear();
    CellEntry *e = m_model->itemFromIndex(m_proxy->mapToSource(idx));
    if(e && e->parent() && e->parent()->parent()) {
        m_view->scene->addRect(e->rect(), QPen(Qt::red, 0));
        m_view->graphicsView->fitInView(e->rect(), Qt::KeepAspectRatio);
    }
}
void lcMainWindow::onDone() { m_view->treeView->expand(m_proxy->index(0, 0)); }