#include "ViewYmlDisplay.h"


ViewYmlDisplay::ViewYmlDisplay(QWidget *parent)
    : QWidget(parent)
{

    Widgets();
    Layouts();
}

void ViewYmlDisplay::Widgets()
{
    twYmlDisplay = new QTreeWidget();
    twYmlDisplay->setColumnCount(3); // Set the number of columns to 3
    QStringList headers;
    headers << "Key" << "Type" << "Value";
    twYmlDisplay->setHeaderLabels(headers);

    teManual = new QTextEdit;
}

void ViewYmlDisplay::Layouts()
{
    QVBoxLayout* vlytMain = new QVBoxLayout;
    vlytMain->setContentsMargins(0,0,0,0);
    {
        spltMain = new QSplitter(Qt::Vertical);
        spltMain->addWidget(twYmlDisplay);
        spltMain->addWidget(teManual);
    }
    vlytMain->addWidget(spltMain);
    setLayout(vlytMain);
}
