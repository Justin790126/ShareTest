#include "ViewGaugeSetupDialog.h"

ViewGaugeSetupDialog::ViewGaugeSetupDialog(QWidget *parent) : QDialog(parent)
{
    setWindowTitle("Gauge Setup");
    UI();
    Connect();

    resize(640,480);
}

QFrame *ViewGaugeSetupDialog::CreateSeparator()
{
    QFrame *separator = new QFrame(this);
    separator->setFrameShape(QFrame::HLine);
    separator->setFrameShadow(QFrame::Sunken);
    return separator;
}

QFrame *ViewGaugeSetupDialog::CreateVerticalSeparator()
{
    QFrame *separator = new QFrame(this);
    separator->setFrameShape(QFrame::VLine);
    separator->setFrameShadow(QFrame::Sunken);
    return separator;
}


void ViewGaugeSetupDialog::UI()
{
    Widgets();
    Layout();
}

void ViewGaugeSetupDialog::Widgets()
{
    chbNeglectWhiteSpace = new QCheckBox("Neglect white space in separation");
    bgDelimiters = new QButtonGroup;
    bgDelimiters->setExclusive(true);

    leOtherDelimiter = new QLineEdit;
    leOtherDelimiter->setPlaceholderText("Enter delimiter");

    btnCancel = new QPushButton("Cancel");
    btnOk = new QPushButton("Ok");
}

void ViewGaugeSetupDialog::Connect()
{

}

QWidget* ViewGaugeSetupDialog::CreatedDelimiterSetupWidget()
{
    QWidget* widResult = new QWidget;
    QVBoxLayout* vlytMain = new QVBoxLayout;
    {
        vlytMain->addWidget(new QLabel("Setup delimiter and preview table content"));
        QHBoxLayout* hlytMain = new QHBoxLayout;
        {
            QGroupBox* gpbDelimiter = new QGroupBox("Delimiter");
            {
                QCheckBox* chbTab = new QCheckBox("Tab (\\t)");
                QCheckBox* chbSemicol = new QCheckBox("Semicolon (;)");
                QCheckBox* chbComma = new QCheckBox("Commna (,)");
                QCheckBox* chbSpace = new QCheckBox("Space ( )");
                QCheckBox* chbOthers = new QCheckBox("Others: ");

                bgDelimiters->addButton(chbTab, 0);
                bgDelimiters->addButton(chbSemicol, 1);
                bgDelimiters->addButton(chbComma, 2);
                bgDelimiters->addButton(chbSpace, 3);
                bgDelimiters->addButton(chbOthers, 4);

                QVBoxLayout* vlytDeli = new QVBoxLayout;
                {
                    vlytDeli->addWidget(chbTab);
                    vlytDeli->addWidget(chbSemicol);
                    vlytDeli->addWidget(chbComma);
                    vlytDeli->addWidget(chbSpace);
                    QHBoxLayout* hlytOtherSep = new QHBoxLayout;
                    {
                        hlytOtherSep->addWidget(chbOthers);
                        hlytOtherSep->addWidget(leOtherDelimiter);
                    }
                    vlytDeli->addLayout(hlytOtherSep);
                }
                

                gpbDelimiter->setLayout(vlytDeli);
            }
            QVBoxLayout* vlytContentSetup = new QVBoxLayout;
            {
                vlytContentSetup->addWidget(chbNeglectWhiteSpace);
            }
            hlytMain->addWidget(gpbDelimiter);
            hlytMain->addLayout(vlytContentSetup);
        }
        vlytMain->addLayout(hlytMain);
        vlytMain->addStretch();


    }
    widResult->setLayout(vlytMain);
    return widResult;
}

void ViewGaugeSetupDialog::Layout()
{
    QWidget* widDelimiter = CreatedDelimiterSetupWidget();
    QTabWidget* tbw = new QTabWidget;
    tbw->addTab(widDelimiter, "Delimiter Setup");

    QVBoxLayout* vlytMain = new QVBoxLayout;
    {
        vlytMain->addWidget(tbw);
        vlytMain->addWidget(CreateSeparator());

        QHBoxLayout* hlytBtns = new QHBoxLayout;
        {
            hlytBtns->addStretch();
            hlytBtns->addWidget(btnCancel);
            hlytBtns->addWidget(btnOk);
        }
        vlytMain->addLayout(hlytBtns);
    }
    this->setLayout(vlytMain);
}
