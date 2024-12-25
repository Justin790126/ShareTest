#include "ViewXmlTab.h"

ViewXmlTab::ViewXmlTab(QWidget *parent) : QWidget(parent)
{
    UI();

}

void ViewXmlTab::Widgets()
{
    QStringList lst = {"Tag", "Text Content", "Attributes", "Value"};
   twXmlViewerLeft = new QTreeWidget(this);
   twXmlViewerLeft->setColumnCount(4);
   twXmlViewerLeft->setHeaderLabels(lst);

   twXmlViewerRight = new QTreeWidget(this);
   twXmlViewerRight->setColumnCount(4);
   twXmlViewerRight->setHeaderLabels(lst);
}

void ViewXmlTab::Layout()
{
    QVBoxLayout *mainLayout = new QVBoxLayout(this);
    mainLayout->setContentsMargins(0, 0, 0, 0);
    {
        // add vertical splitter
        QSplitter *splitter = new QSplitter(this);
        splitter->setOrientation(Qt::Vertical);
        
        QWidget* widGraph = new QWidget(splitter);
        {
            // add horizontal splitter
            QSplitter *splitterH = new QSplitter(widGraph);
            splitterH->setOrientation(Qt::Horizontal);
            // add graph view
           splitterH->addWidget(new QLabel("control"));
           splitterH->addWidget(new QLabel("Wafer map"));
        }
        QWidget* widXmlView = new QWidget(splitter);
        {
            QHBoxLayout* hlytXmlTree = new QHBoxLayout(widXmlView);
            hlytXmlTree->setContentsMargins(0, 0, 0, 0);
            {
                QVBoxLayout* vlytLeftTree = new QVBoxLayout();
                vlytLeftTree->setContentsMargins(0, 0, 0, 0);
                {
                    // add QComboBox
                    QComboBox* cbbXmlLeftSrc = new QComboBox;
                    vlytLeftTree->addWidget(cbbXmlLeftSrc);
                    vlytLeftTree->addWidget(twXmlViewerLeft);
                }
                QVBoxLayout* vlytRightTree = new QVBoxLayout();
                vlytRightTree->setContentsMargins(0, 0, 0, 0);
                {
                    // add QComboBox
                    QComboBox* cbbXmlRightSrc = new QComboBox;
                    vlytRightTree->addWidget(cbbXmlRightSrc);
                    vlytRightTree->addWidget(twXmlViewerRight);
                }
                hlytXmlTree->addLayout(vlytLeftTree);
                hlytXmlTree->addLayout(vlytRightTree);
            }
        }
        QWidget* widDiffSummary = new QWidget(splitter);
        {
            QVBoxLayout* vlytDiffSummary = new QVBoxLayout(widDiffSummary);
            lwDiffSummary = new QListWidget(widDiffSummary);
            vlytDiffSummary->addWidget(new QLabel("Difference Summary"));
            vlytDiffSummary->addWidget(lwDiffSummary);
        }
        mainLayout->addWidget(splitter);
        splitter->setSizes({200, 600, 100});
    }
    this->setLayout(mainLayout);
}

void ViewXmlTab::UI()
{
    Widgets();
    Layout();
}