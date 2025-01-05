#ifndef VIEW_PRODUCT_DIALOG_H
#define VIEW_PRODUCT_DIALOG_H

#include <QtGui>
#include <iostream>

using namespace std;


class ProdcutTreeItem : public QTreeWidgetItem
{
public:
    ProdcutTreeItem(const QStringList &strings, QTreeWidget *parent = nullptr)
        : QTreeWidgetItem(parent, strings) {}

    bool operator<(const QTreeWidgetItem &other) const override
    {
        // Get the data to compare from the items
        QString thisData = text(0); // Assuming data is in the first column
        QString otherData = other.text(0);

        // Perform custom comparison
        // Here's a simple example: compare alphabetically, case-insensitive
        return QString::compare(thisData, otherData, Qt::CaseInsensitive) < 0;
    }
};

class ViewAddProductDialog : public QDialog
{
    Q_OBJECT
public:
    ViewAddProductDialog(QWidget *parent = NULL);
    ~ViewAddProductDialog() = default;


private:
    void Widgets();
    void Layout();
    void Connect();
private:
    QLineEdit *leProductName = NULL;
    QLineEdit *leWfrLen = NULL;
    QLineEdit *leWfrSize = NULL;
    QLineEdit *leWfrOffsetX = NULL;
    QLineEdit *leWfrOffsetY = NULL;
    QPushButton *btnAdd = NULL;
    QPushButton *btnCancel = NULL;
};

class ViewProductDialog : public QDialog
{
    Q_OBJECT
public:
    ViewProductDialog(QWidget *parent = NULL);
    ~ViewProductDialog() = default;

    QFrame *CreateSeparator();

    QPushButton *GetAddButton() { return btnAdd; }

private:
    QPushButton *btnOk = NULL;
    QPushButton *btnCancel = NULL;
    QPushButton *btnAdd = NULL;
    QPushButton *btnDel = NULL;
    QPushButton *btnOrderUp = NULL;
    QPushButton *btnOrderDown = NULL;
    QTreeWidget *twProductList = NULL;

    QLineEdit *leSearchBar = NULL;

    void Widgets();
    void Layout();
    void UI();

private slots:
    void handleBtnOkPressed();
    void handleBtnCancelPressed();
};

#endif /* VIEW_PRODUCT_DIALOG_H */