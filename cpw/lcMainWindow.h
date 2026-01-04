#ifndef LCMAINWINDOW_H
#define LCMAINWINDOW_H
#include <QObject>
#include <QModelIndex>
#include "ViewMainWindow.h"
#include "ModelCellProfile.h"

class lcMainWindow : public QObject {
    Q_OBJECT
public:
    lcMainWindow();
    void loadProfile(const QString &fileName);
private slots:
    void onCellClicked(const QModelIndex &index);
    void onLoadingFinished();
private:
    ViewMainWindow *m_view;
    ModelCellProfile *m_model;
};
#endif