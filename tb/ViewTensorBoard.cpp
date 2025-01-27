#include "ViewTensorBoard.h"

ViewTensorBoard::ViewTensorBoard(QWidget *parent)
    : QWidget(parent)
{
    UI();
}

void ViewTensorBoard::widgets()
{
    tbwMain = new QTabWidget(this);
    QLabel* lblTitle = new QLabel("TensorBoard");
    // set font size
    lblTitle->setFont(QFont("Arial", 18, QFont::Bold));
    tbwMain->setCornerWidget(lblTitle, Qt::TopLeftCorner);

    tbwMain->addTab(new QWidget(), tr("Dashboard"));
    tbwMain->addTab(new QWidget(), tr("Runs"));
    tbwMain->addTab(new QWidget(), tr("Events"));


}

void ViewTensorBoard::layouts()
{
    QVBoxLayout *vlytMain = new QVBoxLayout(this);
    {
        vlytMain->addWidget(tbwMain);
    }
    this->setLayout(vlytMain);
}

void ViewTensorBoard::UI()
{
    widgets();
    layouts();

    // Set up the window
    setWindowTitle(tr("Iht TensorBoard"));
    setMinimumSize(800, 600);
    
}