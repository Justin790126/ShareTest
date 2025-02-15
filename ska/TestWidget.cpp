#include "TestWidget.h"

TestWidget::TestWidget(QWidget *parent)
    : QWidget(parent)
{
    UI();
}

void TestWidget::Connect()
{
    // connect textedit with SIGNAL/SLOT
}

void TestWidget::UI()
{
    widgets();
    layouts();
    Connect();
}

void TestWidget::layouts()
{
    QVBoxLayout *layout = new QVBoxLayout;
    layout->addWidget(teOutput);
    QHBoxLayout *hlyt = new QHBoxLayout;
    {
        hlyt->addWidget(btnConnect);
        hlyt->addWidget(btnSend);
        hlyt->addWidget(btnSendContourMaked);
    }
    layout->addLayout(hlyt);
    setLayout(layout);
}


void TestWidget::widgets()
{
    btnConnect = new QPushButton("Connect");
    btnSend = new QPushButton("Server shutdown");
    btnSendContourMaked = new QPushButton("Contour Make");
    teOutput = new QTextEdit;
    teOutput->setReadOnly(true);
}
