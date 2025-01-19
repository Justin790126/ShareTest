#include "ViewXmlTab.h"

ViewXmlTab::ViewXmlTab(QWidget *parent) : QWidget(parent)
{
    UI();
    Connect();
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

    // use style sheet to change background color, text color on selection
    twXmlViewerLeft->setStyleSheet("QTreeWidget::item:selected { background-color: blue; color: white; }");
    twXmlViewerRight->setStyleSheet("QTreeWidget::item:selected { background-color: blue; color: white; }");
}

void ViewXmlTab::Connect()
{
    connect(twXmlViewerLeft->verticalScrollBar(), SIGNAL(valueChanged(int)),
            twXmlViewerRight->verticalScrollBar(), SLOT(setValue(int)));
    connect(twXmlViewerRight->verticalScrollBar(), SIGNAL(valueChanged(int)),
            twXmlViewerLeft->verticalScrollBar(), SLOT(setValue(int)));

    // copy selected text
    connect(twXmlViewerLeft, SIGNAL(itemSelectionChanged()), this, SLOT(handleClipItemToClipBoard()));
    connect(twXmlViewerRight, SIGNAL(itemSelectionChanged()), this, SLOT(handleClipItemToClipBoard()));
}

void ViewXmlTab::handleClipItemToClipBoard()
{
    // copy a row of data to clipboard
    QTreeWidget *tw = (QTreeWidget *)QObject::sender();
    QTreeWidgetItem *item = tw->selectedItems().first();
    if (item)
    {
        QClipboard *clipboard = QApplication::clipboard();
        QStringList rowData;

        // Get data from all columns of the selected item
        for (int col = 0; col < item->columnCount(); ++col)
        {
            rowData << item->text(col);
        }

        clipboard->setText(rowData.join("\t")); // Use tab as delimiter
        qDebug() << "Copied row data to clipboard:";
        qDebug() << rowData.join("\t");
    }
}

void ViewXmlTab::Layout()
{
    QVBoxLayout *mainLayout = new QVBoxLayout(this);
    mainLayout->setContentsMargins(0, 0, 0, 0);
    {
        // add vertical splitter
        QSplitter *splitter = new QSplitter(this);
        splitter->setOrientation(Qt::Vertical);

        QWidget *widGraph = new QWidget(splitter);
        {
            // add horizontal splitter
            QSplitter *splitterH = new QSplitter(widGraph);
            splitterH->setOrientation(Qt::Horizontal);
            // add graph view
            splitterH->addWidget(new QLabel("control"));
            splitterH->addWidget(new QLabel("Wafer map"));
        }
        QWidget *widXmlView = new QWidget(splitter);
        {
            QHBoxLayout *hlytXmlTree = new QHBoxLayout(widXmlView);
            hlytXmlTree->setContentsMargins(0, 0, 0, 0);
            {
                QVBoxLayout *vlytLeftTree = new QVBoxLayout();
                vlytLeftTree->setContentsMargins(0, 0, 0, 0);
                {
                    // add QComboBox
                    QComboBox *cbbXmlLeftSrc = new QComboBox;
                    vlytLeftTree->addWidget(cbbXmlLeftSrc);
                    vlytLeftTree->addWidget(twXmlViewerLeft);
                }
                QVBoxLayout *vlytRightTree = new QVBoxLayout();
                vlytRightTree->setContentsMargins(0, 0, 0, 0);
                {
                    // add QComboBox
                    QComboBox *cbbXmlRightSrc = new QComboBox;
                    vlytRightTree->addWidget(cbbXmlRightSrc);
                    vlytRightTree->addWidget(twXmlViewerRight);
                }
                hlytXmlTree->addLayout(vlytLeftTree);
                hlytXmlTree->addLayout(vlytRightTree);
            }
        }
        QWidget *widDiffSummary = new QWidget(splitter);
        {
            QVBoxLayout *vlytDiffSummary = new QVBoxLayout(widDiffSummary);
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