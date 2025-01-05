#include "ViewProductDialog.h"

/*

    Add Product Dialog

*/

ViewAddProductDialog::ViewAddProductDialog(QWidget *parent) : QDialog(parent)
{
    // Set window title
    setWindowTitle("Add New Product");

    // Create widgets
    Widgets();

    // Create layout
    Layout();

    // Connect signals and slots
    Connect();
}

void ViewAddProductDialog::Widgets()
{
    leProductName = new QLineEdit();
    leWfrLen = new QLineEdit();
    leWfrSize = new QLineEdit();
    leWfrOffsetX = new QLineEdit();
    leWfrOffsetY = new QLineEdit();
    btnAdd = new QPushButton("Add");
    btnCancel = new QPushButton("Cancel");
}

void ViewAddProductDialog::Layout()
{
    QVBoxLayout* vlytMain = new QVBoxLayout();
    {
        QFormLayout* fmltMain = new QFormLayout();
        {
            fmltMain->addRow("Product Name:", leProductName);
            fmltMain->addRow("Wafer Length:", leWfrLen);
            fmltMain->addRow("Wafer Size:", leWfrSize);
            fmltMain->addRow("Wafer Offset X:", leWfrOffsetX);
            fmltMain->addRow("Wafer Offset Y:", leWfrOffsetY);
        }
        QHBoxLayout *hlytBtns = new QHBoxLayout;
        {
            hlytBtns->addStretch();
            hlytBtns->addWidget(btnCancel);
            hlytBtns->addWidget(btnAdd);
        }
        vlytMain->addLayout(fmltMain);
        vlytMain->addLayout(hlytBtns);
    }
    setLayout(vlytMain);
}

void ViewAddProductDialog::Connect()
{
    // connect with SIGNAL , SLOT
    connect(btnAdd, SIGNAL(clicked()), this, SLOT(accept()));
    connect(btnCancel, SIGNAL(clicked()), this, SLOT(close()));
}

/*

    Main Product Dialog

 */
ViewProductDialog::ViewProductDialog(QWidget *parent) : QDialog(parent)
{
    UI();
}

void ViewProductDialog::UI()
{
    Widgets();
    Layout();
}


void ViewProductDialog::Widgets()
{
    btnOk = new QPushButton("Ok");
    btnCancel = new QPushButton("Cancel");
    btnAdd = new QPushButton("Add");
    btnDel = new QPushButton("Delete");
    btnOrderUp = new QPushButton("Up");
    btnOrderDown = new QPushButton("Down");
    leSearchBar = new QLineEdit();
    leSearchBar->setPlaceholderText("Filter by Product Name");

    twProductList = new QTreeWidget();
    twProductList->setColumnCount(5);
    twProductList->setHeaderLabels(QStringList() << "Product Name" << "Wafer Length" << "Wafer Size" << "Wafer Offset X" << "Wafer Offset Y");
}

void ViewProductDialog::Layout()
{
    QVBoxLayout *vlytMain = new QVBoxLayout();
    {
        QHBoxLayout *hlytLeftRight = new QHBoxLayout();
        hlytLeftRight->setContentsMargins(0, 0, 0, 0);
        {
            QVBoxLayout *vlytProduct = new QVBoxLayout();
            vlytProduct->setContentsMargins(0, 0, 0, 0);
            {
                QHBoxLayout *hlytSearchBar = new QHBoxLayout();
                {
                    hlytSearchBar->addWidget(new QLabel("Filter by <b>product name</b>"));
                    hlytSearchBar->addWidget(leSearchBar);
                }
                vlytProduct->addLayout(hlytSearchBar);
                vlytProduct->addWidget(twProductList);
            }
            hlytLeftRight->addLayout(vlytProduct);
        }
        QHBoxLayout *hlytBtns = new QHBoxLayout();
        {
            hlytBtns->addStretch();
            hlytBtns->addWidget(btnCancel);
            hlytBtns->addWidget(btnOk);
        }
        QHBoxLayout *hlytToolBtn = new QHBoxLayout();
        {
            hlytToolBtn->addWidget(btnOrderDown);
            hlytToolBtn->addWidget(btnOrderUp);
            hlytToolBtn->addWidget(btnDel);
            hlytToolBtn->addWidget(btnAdd);
            hlytToolBtn->addStretch();
        }
        vlytMain->addLayout(hlytLeftRight, 8);
        vlytMain->addLayout(hlytToolBtn, 1);
        vlytMain->addLayout(hlytBtns, 1);
    }
    this->setLayout(vlytMain);
    this->resize(510, 680);
}

QFrame *ViewProductDialog::CreateSeparator()
{
    QFrame *separator = new QFrame(this);
    separator->setFrameShape(QFrame::HLine);
    separator->setFrameShadow(QFrame::Sunken);
    return separator;
}

void ViewProductDialog::handleBtnCancelPressed()
{
    reject();
}

void ViewProductDialog::handleBtnOkPressed()
{
    accept();
}