#include "ViewLytMergeDialog.h"

/*

    ViewLytMergeDialog

 */
ViewLytMergeDialog::ViewLytMergeDialog(QWidget *parent) : QDialog(parent)
{
    UI();
    Connect();

    resize(1280, 480);
}

void ViewLytMergeDialog::Connect()
{
    connect(btnMergeStepMapping, SIGNAL(clicked()), this, SLOT(handleToggleLyrMappingWidget()));
    connect(btnMergeStepOffsetRot, SIGNAL(clicked()), this, SLOT(handleToggleLyrOffsetRotWidget()));
    // connect(btnOk, SIGNAL(clicked()), this, SLOT(accept()));
    // connect(btnCancel, SIGNAL(clicked()), this, SLOT(reject()));
    // connect(btnLoad, SIGNAL(clicked()), this, SIGNAL(loadConfig()));
    // connect(btnSave, SIGNAL(clicked()), this, SIGNAL(saveConfig()));
    // connect(btnAdd, SIGNAL(clicked()), this, SIGNAL(addNewProduct()));
    // connect(btnDel, SIGNAL(clicked()), this, SIGNAL(delSelProduct()));
    // connect(shtDel, SIGNAL(activated()), this, SIGNAL(delSelProduct()));
    // connect(leSearchBar, SIGNAL(textChanged(const QString &)), this, SIGNAL(searchKeyChanged(const QString &)));
}

void ViewLytMergeDialog::handleToggleLyrMappingWidget()
{
    bool vis = !widLytMapping->isVisible();
    widLytMapping->setVisible(vis);
}

void ViewLytMergeDialog::handleToggleLyrOffsetRotWidget()
{
    bool vis =!widLytOffsetRot->isVisible();
    widLytOffsetRot->setVisible(vis);
}

void ViewLytMergeDialog::UI()
{
    Widgets();
    Layout();
}

void ViewLytMergeDialog::Widgets()
{
    pgMergeState = new QProgressBar;
    btnCancel = new QPushButton("Cancel");
    btnLoad = new QPushButton("Load Layouts");
    twLytPreLoad = new QTreeWidget();
    twLytPreLoad->setColumnCount(2);
    twLytPreLoad->setHeaderLabels(QStringList() << "Layout Name" << "Layer");

    btnMergeStepMapping = new QPushButton();
    QIcon rightArrowIcon = QApplication::style()->standardIcon(QStyle::SP_ArrowRight);
    btnMergeStepMapping->setIcon(rightArrowIcon);

    tbLyrMapping = new QTableWidget();
    tbLyrMapping->setColumnCount(2);
    tbLyrMapping->setHorizontalHeaderLabels(QStringList() << "Possible Layers" << "Renamed Layers");

    widLytMapping = new QWidget;
    widLytMapping->setVisible(false);

    twLytOffsetRotation = new QTreeWidget();
    twLytOffsetRotation->setColumnCount(3);
    twLytOffsetRotation->setHeaderLabels(QStringList() << "Offset X(um)" << "Offset Y(um)" << "Rotation");

    btnMergeStepOffsetRot = new QPushButton();
    btnMergeStepOffsetRot->setIcon(rightArrowIcon);

    widLytOffsetRot = new QWidget;
    widLytOffsetRot->setVisible(false);

    tlbtnMerge = new QToolButton();
    tlbtnMerge->setText("Merge");
    tlbtnMerge->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);
    tlbtnMerge->setIcon(rightArrowIcon);

}

void ViewLytMergeDialog::Layout()
{
    QVBoxLayout *vlytMain = new QVBoxLayout();
    vlytMain->addWidget(pgMergeState);
    vlytMain->addWidget(CreateSeparator());
    QHBoxLayout *hlytMergeFlow = new QHBoxLayout;
    {
        QVBoxLayout *vlytPreproc = new QVBoxLayout;
        {
            QHBoxLayout *hlytBtns = new QHBoxLayout;
            {
                hlytBtns->addWidget(btnLoad);
                hlytBtns->addStretch();
            }
            vlytPreproc->addLayout(hlytBtns);
            vlytPreproc->addWidget(twLytPreLoad);
        }
        hlytMergeFlow->addLayout(vlytPreproc);

        QVBoxLayout *vlytMapping = new QVBoxLayout;
        {
            vlytMapping->addStretch();
            vlytMapping->addWidget(btnMergeStepMapping);
            vlytMapping->addStretch();
        }
        hlytMergeFlow->addLayout(vlytMapping);

        {
            QHBoxLayout *hlytLytMapping = new QHBoxLayout;
            {
                QVBoxLayout *vlytLytMapping = new QVBoxLayout;
                {
                    vlytLytMapping->addWidget(new QLabel("Rename layers"));
                    vlytLytMapping->addWidget(tbLyrMapping);
                }
                QVBoxLayout *vlytOffsetRot = new QVBoxLayout;
                {
                    vlytOffsetRot->addStretch();
                    vlytOffsetRot->addWidget(btnMergeStepOffsetRot);
                    vlytOffsetRot->addStretch();
                }
                hlytLytMapping->addLayout(vlytLytMapping);
                hlytLytMapping->addLayout(vlytOffsetRot);
            }

            widLytMapping->setLayout(hlytLytMapping);
        }
        hlytMergeFlow->addWidget(widLytMapping);

        {
            QHBoxLayout* hlytOffsetRot = new QHBoxLayout;
            {
                QVBoxLayout* vlytOffsetRot = new QVBoxLayout;
                {
                    vlytOffsetRot->addWidget(new QLabel("Setup offset, rotation"));
                    vlytOffsetRot->addWidget(twLytOffsetRotation);
                }
                QVBoxLayout* vlytMerge = new QVBoxLayout;
                {
                    vlytMerge->addStretch();
                    vlytMerge->addWidget(tlbtnMerge);
                    vlytMerge->addStretch();
                }
                hlytOffsetRot->addLayout(vlytOffsetRot);
                hlytOffsetRot->addLayout(vlytMerge);
            }
            widLytOffsetRot->setLayout(hlytOffsetRot);
        }
        hlytMergeFlow->addWidget(widLytOffsetRot);
    }
    vlytMain->addLayout(hlytMergeFlow);
    vlytMain->addWidget(CreateSeparator());
    QHBoxLayout *hlytBtns = new QHBoxLayout;
    {
        hlytBtns->addStretch();
        hlytBtns->addWidget(btnCancel);
    }

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
