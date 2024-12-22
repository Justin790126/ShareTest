#include "ViewXmlTab.h"

ViewXmlTab::ViewXmlTab(QWidget *parent) : QWidget(parent)
{
    UI();

}

void ViewXmlTab::Widgets()
{
   twXmlViewer = new QTreeWidget(this);
   twXmlViewer->setColumnCount(4);
   QStringList lst = {"Tag", "Attributes", "Value", "Text Content"};
   twXmlViewer->setHeaderLabels(lst);
}

void ViewXmlTab::Layout()
{
    QVBoxLayout *mainLayout = new QVBoxLayout(this);
    mainLayout->addWidget(twXmlViewer);
    this->setLayout(mainLayout);
}

void ViewXmlTab::UI()
{
    Widgets();
    Layout();
}