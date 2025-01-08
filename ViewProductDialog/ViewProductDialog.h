#ifndef VIEW_PRODUCT_DIALOG_H
#define VIEW_PRODUCT_DIALOG_H

#include <QtGui>
#include <iostream>
#include "ModelOvlConf.h"

using namespace std;


class ProductTreeItem : public QTreeWidgetItem
{
public:
    ProductTreeItem(QStringList &strings, QTreeWidget *parent = nullptr)
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
    void SetProductInfo(OvlProductInfo* pdInfo) { m_pdInfo = pdInfo; }
    OvlProductInfo* GetProductInfo() { return m_pdInfo; }
private:
    OvlProductInfo* m_pdInfo = NULL;
};

class ViewAddProductDialog : public QDialog
{
    Q_OBJECT
public:
    ViewAddProductDialog(QWidget *parent = NULL);
    ~ViewAddProductDialog() = default;

    QLineEdit *GetProductNameLineEdit() { return leProductName; }
    QLineEdit *GetDieWLineEdit() { return leDieW; }
    QLineEdit *GetDieHLineEdit() { return leDieH; }
    QLineEdit *GetDieOffsetXLineEdit() { return leDieOffsetX; }
    QLineEdit *GetDieOffsetYLineEdit() { return leDieOffsetY; }

private:
    void Widgets();
    void Layout();
    void Connect();
private:
    QLineEdit *leProductName = NULL;
    QLineEdit *leDieW = NULL;
    QLineEdit *leDieH = NULL;
    QLineEdit *leDieOffsetX = NULL;
    QLineEdit *leDieOffsetY = NULL;
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
    QFrame* CreateVerticalSeparator();

    QPushButton *GetAddButton() { return btnAdd; }
    QTreeWidget *GetProductTreeWidget() { return twProductList; }

signals:
    void loadConfig();
    void saveConfig();
    void addNewProduct();
    void delSelProduct();

private:
    QPushButton *btnOk = NULL;
    QPushButton *btnCancel = NULL;
    QPushButton *btnAdd = NULL;
    QPushButton *btnDel = NULL;
    QPushButton *btnLoad = NULL;
    QPushButton *btnSave = NULL;
    QTreeWidget *twProductList = NULL;

    QShortcut* shtDel = NULL;

    QLineEdit *leSearchBar = NULL;

    void Widgets();
    void Layout();
    void UI();
    void Connect();

private slots:
    void handleBtnOkPressed();
    void handleBtnCancelPressed();
};

#endif /* VIEW_PRODUCT_DIALOG_H */