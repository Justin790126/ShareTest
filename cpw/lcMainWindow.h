#ifndef LCMAINWINDOW_H
#define LCMAINWINDOW_H
#include <QObject>
#include <QSortFilterProxyModel>
#include "ViewMainWindow.h"
#include "ModelCellProfile.h"

class RecursiveProxy : public QSortFilterProxyModel {
protected:
    bool filterAcceptsRow(int r, const QModelIndex &p) const {
        QModelIndex idx = sourceModel()->index(r, 0, p);
        if (sourceModel()->data(idx).toString().contains(filterRegExp())) return true;
        for (int i = 0; i < sourceModel()->rowCount(idx); ++i) 
            if (filterAcceptsRow(i, idx)) return true;
        return false;
    }
};

class lcMainWindow : public QObject {
    Q_OBJECT
public:
    explicit lcMainWindow(QObject *parent = 0);
    void loadProfile(const QString &fileName);
private slots:
    void onSearch(const QString &t);
    void onClick(const QModelIndex &i);
    void onDone();
private:
    ViewMainWindow *m_view;
    ModelCellProfile *m_model;
    RecursiveProxy *m_proxy;
};
#endif