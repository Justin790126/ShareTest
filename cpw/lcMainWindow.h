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
    virtual ~lcMainWindow();
    void generateAndLoad(int count);

private slots:
    void onCellClicked(const QModelIndex &index);

private:
    ViewMainWindow   *m_view;
    ModelCellProfile *m_model;
};

#endif // LCMAINWINDOW_H