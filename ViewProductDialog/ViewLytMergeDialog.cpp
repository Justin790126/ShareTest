#include "ViewLytMergeDialog.h"

/*

    ViewLytMergeDialog

 */
ViewLytMergeDialog::ViewLytMergeDialog(QWidget *parent) : QDialog(parent)
{
    UI();
    Connect();

    resize(640, 480);
}

void ViewLytMergeDialog::Connect()
{

}

void ViewLytMergeDialog::UI()
{
    Widgets();
    Layout();
}

void ViewLytMergeDialog::Widgets()
{
    pgMergeState = new QProgressBar;
    btnMerge = new QPushButton("Merge");
    btnCancel = new QPushButton("Cancel");
    btnLoad = new QPushButton("Load Layouts");
    // btnPreview = new QPushButton("Preview");
    // btnPreview->setVisible(false);
    twMergeSetting = new QTreeWidget();
    twMergeSetting->setColumnCount(6);
    twMergeSetting->setHeaderLabels(QStringList() << "Layout Name" << "Layers"<< "Renamed Layers" << "Offset X(um)" << "Offset Y(um)" << "Rotation");
}

void ViewLytMergeDialog::Layout()
{
    QVBoxLayout *vlytMain = new QVBoxLayout();
    
    QHBoxLayout* hlytHeadBtns = new QHBoxLayout;
    {
        hlytHeadBtns->addWidget(btnLoad);
        hlytHeadBtns->addStretch();
    }
    QHBoxLayout *hlytMergeFlow = new QHBoxLayout;
    {
       hlytMergeFlow->addWidget(twMergeSetting);
    }
    QHBoxLayout *hlytBtns = new QHBoxLayout;
    {
        hlytBtns->addStretch();
        hlytBtns->addWidget(btnCancel);
        hlytBtns->addWidget(btnMerge);
    }

    vlytMain->addWidget(pgMergeState);
    vlytMain->addLayout(hlytHeadBtns);
    vlytMain->addWidget(CreateSeparator());
    vlytMain->addLayout(hlytMergeFlow);
    vlytMain->addWidget(CreateSeparator());
    vlytMain->addLayout(hlytBtns);
    this->setLayout(vlytMain);
}

QFrame *ViewLytMergeDialog::CreateSeparator()
{
    QFrame *separator = new QFrame(this);
    separator->setFrameShape(QFrame::HLine);
    separator->setFrameShadow(QFrame::Sunken);
    return separator;
}

QFrame *ViewLytMergeDialog::CreateVerticalSeparator()
{
    QFrame *separator = new QFrame(this);
    separator->setFrameShape(QFrame::VLine);
    separator->setFrameShadow(QFrame::Sunken);
    return separator;
}
