#ifndef LCMAINWINDOW_H
#define LCMAINWINDOW_H
#include <QObject>
#include <QModelIndex>
#include <QSortFilterProxyModel>
#include "ViewMainWindow.h"
#include "ModelCellProfile.h"

// 為了實現無腦搜尋，我們稍微自定義 ProxyModel
class RecursiveFilterProxy : public QSortFilterProxyModel {
protected:
    // 關鍵：如果子節點符合，父節點也要顯示；如果父節點符合，子節點也要顯示
    bool filterAcceptsRow(int source_row, const QModelIndex &source_parent) const {
        if (filterRegExp().isEmpty()) return true;
        
        // 檢查當前節點是否符合
        QModelIndex index = sourceModel()->index(source_row, 0, source_parent);
        if (sourceModel()->data(index).toString().contains(filterRegExp().pattern(), Qt::CaseInsensitive))
            return true;

        // 遞迴檢查：如果它是分類節點 (Top/All Cells)，且其下有子節點符合，則顯示分類
        int rows = sourceModel()->rowCount(index);
        for (int i = 0; i < rows; ++i) {
            if (filterAcceptsRow(i, index)) return true;
        }
        return false;
    }
};

class lcMainWindow : public QObject {
    Q_OBJECT
public:
    lcMainWindow();
    void loadProfile(const QString &fileName);
private slots:
    void onCellClicked(const QModelIndex &proxyIdx);
    void onLoadingFinished();
    void onSearchTextChanged(const QString &text);
private:
    ViewMainWindow *m_view;
    ModelCellProfile *m_model;
    RecursiveFilterProxy *m_proxy; // 使用強化版過濾器
};
#endif