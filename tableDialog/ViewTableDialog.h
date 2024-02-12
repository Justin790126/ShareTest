#ifndef ViewTableDialog_H
#define ViewTableDialog_H

#include <QDialog>
#include <QPushButton>
#include <QLabel>
#include <QTreeView>
#include <QScrollArea>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QStandardItemModel>
#include "ViewColEdit.h"

class ViewTableDialog : public QDialog
{
    Q_OBJECT

public:
    ViewTableDialog(QWidget *parent = NULL);

private slots:
    void applyChanges();

private:

    QPushButton *okButton;
    QPushButton *cancelButton;
    QPushButton *applyButton;
    QTreeView *treeView;
};

#endif // ViewTableDialog_H
