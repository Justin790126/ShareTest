#ifndef VIEW_LYT_MERGE_DIALOG_H
#define VIEW_LYT_MERGE_DIALOG_H

#include <QtGui>
#include <iostream>

class ViewLytMergeDialog : public QDialog
{
    Q_OBJECT
public:
    ViewLytMergeDialog(QWidget *parent = NULL);
    ~ViewLytMergeDialog() = default;

    QFrame *CreateSeparator();
    QFrame *CreateVerticalSeparator();

private slots:
    void handleToggleLyrMappingWidget();
    void handleToggleLyrOffsetRotWidget();

private:
    QPushButton *btnOk = NULL;
    QPushButton *btnCancel = NULL;
    QPushButton *btnLoad = NULL;
    QToolButton* tlbtnMerge = NULL;
    QPushButton *btnMergeStepMapping = NULL;
    QPushButton *btnMergeStepOffsetRot = NULL;

    QTreeWidget *twLytPreLoad = NULL;
    QWidget *widLytMapping = NULL;
    QTableWidget *tbLyrMapping = NULL;
    QWidget* widLytOffsetRot = NULL;
    QTreeWidget *twLytOffsetRotation = NULL;
    QLineEdit *leLytOutName = NULL;

    QProgressBar *pgMergeState = NULL;

    void Widgets();
    void Layout();
    void UI();
    void Connect();
};

#endif /* VIEW_LYT_MERGE_DIALOG_H */