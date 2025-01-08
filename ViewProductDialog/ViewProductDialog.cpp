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
    leDieW = new QLineEdit();
    leDieH = new QLineEdit();
    leDieOffsetX = new QLineEdit();
    leDieOffsetY = new QLineEdit();
    btnAdd = new QPushButton("Add");
    btnCancel = new QPushButton("Cancel");
}

void ViewAddProductDialog::Layout()
{
    QVBoxLayout *vlytMain = new QVBoxLayout();
    {
        QFormLayout *fmltMain = new QFormLayout();
        {
            fmltMain->addRow("Product Name:", leProductName);
            fmltMain->addRow("Field Width(um):", leDieW);
            fmltMain->addRow("Field Height(um):", leDieH);
            fmltMain->addRow("Offset X(um):", leDieOffsetX);
            fmltMain->addRow("Offset Y(um):", leDieOffsetY);
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
    connect(btnCancel, SIGNAL(clicked()), this, SLOT(reject()));
}

/*

    Main Product Dialog

 */
ViewProductDialog::ViewProductDialog(QWidget *parent) : QDialog(parent)
{
    UI();
    Connect();
}

void ViewProductDialog::Connect()
{
    connect(btnOk, SIGNAL(clicked()), this, SLOT(accept()));
    connect(btnCancel, SIGNAL(clicked()), this, SLOT(reject()));
    connect(btnLoad, SIGNAL(clicked()), this, SIGNAL(loadConfig()));
    connect(btnSave, SIGNAL(clicked()), this, SIGNAL(saveConfig()));
    connect(btnAdd, SIGNAL(clicked()), this, SIGNAL(addNewProduct()));
    connect(btnDel, SIGNAL(clicked()), this, SIGNAL(delSelProduct()));
    connect(shtDel, SIGNAL(activated()), this, SIGNAL(delSelProduct()));
    connect(leSearchBar, SIGNAL(textChanged(const QString &)), this, SIGNAL(searchKeyChanged(const QString &)));
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
    shtDel = new QShortcut(QKeySequence::Delete, this);
    btnLoad = new QPushButton("Load");
    btnSave = new QPushButton("Save");
    leSearchBar = new QLineEdit();
    leSearchBar->setPlaceholderText("Filter by Product Name");

    twProductList = new QTreeWidget();
    twProductList->setColumnCount(5);
    twProductList->setHeaderLabels(QStringList() << "Product Name" << "Field Width(um)" << "Field Length(um)" << "Offset X(um)" << "Offset Y(um)");
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
            hlytToolBtn->addWidget(btnAdd);
            hlytToolBtn->addWidget(btnDel);
            hlytToolBtn->addWidget(CreateVerticalSeparator());
            hlytToolBtn->addWidget(btnLoad);
            hlytToolBtn->addWidget(btnSave);
            hlytToolBtn->addStretch(5);
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

QFrame *ViewProductDialog::CreateVerticalSeparator()
{
    QFrame *separator = new QFrame(this);
    separator->setFrameShape(QFrame::VLine);
    separator->setFrameShadow(QFrame::Sunken);
    return separator;
}
