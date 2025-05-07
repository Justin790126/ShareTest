#ifndef VIEW_LYT_MERGE_DIALOG_H
#define VIEW_LYT_MERGE_DIALOG_H

#include <QtGui>
#include <iostream>

/*
    Features TODO:
        1. Change dbu
        2. Merge layout
        3. Preview merge result with bbox
        4. Extract bbox column
*/

class ViewLytMergeDialog : public QDialog
{
    Q_OBJECT
public:
    ViewLytMergeDialog(QWidget *parent = NULL);
    ~ViewLytMergeDialog() = default;

    QFrame *CreateSeparator();
    QFrame *CreateVerticalSeparator();

private slots:

private:
    QPushButton *btnMerge = NULL;
    QPushButton *btnPreview = NULL;
    QPushButton *btnCancel = NULL;
    QPushButton *btnLoad = NULL;
   
    QTreeWidget *twMergeSetting = NULL;
    QProgressBar *pgMergeState = NULL;

    void Widgets();
    void Layout();
    void UI();
    void Connect();
};

#endif /* VIEW_LYT_MERGE_DIALOG_H */