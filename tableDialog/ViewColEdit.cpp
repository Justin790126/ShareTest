#include "ViewColEdit.h"

ViewColEdit::ViewColEdit(QWidget *parent)
    : QScrollArea(parent)
{
    setWidgetResizable(true);
    Widgets();
    Layout();
}


void ViewColEdit::Widgets()
{
    widMain = new QWidget;
    lwAllCols = new QListWidget;
    QStyle *style = QApplication::style();
    // Get the left arrow icon
    QIcon icon = style->standardIcon(QStyle::SP_ArrowLeft);
    btnArrLeft = new QPushButton;
    btnArrLeft->setIcon(icon);
    icon = style->standardIcon(QStyle::SP_ArrowRight);
    btnArrRight = new QPushButton;
    btnArrRight->setIcon(icon);
    lwDesireCols = new QListWidget;

    setWidget(widMain);
}

void ViewColEdit::Layout()
{
    lytMain = new QHBoxLayout(widMain);
    lytMain->setContentsMargins(5, 0, 5, 0);
    lytMain->addWidget(lwAllCols);
    QVBoxLayout *lytBtns = new QVBoxLayout;
    {
        lytBtns->addWidget(btnArrLeft);
        lytBtns->addWidget(btnArrRight);
    }
    lytMain->addLayout(lytBtns);
    lytMain->addWidget(lwDesireCols);
}