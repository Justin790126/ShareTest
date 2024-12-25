#include "lcXml.h"

LcXml::LcXml(QWidget *parent) : QWidget(parent)
{
    m_vtXmlTab = new ViewXmlTab(this);
    // Add UI elements here
    resize(640, 480);

    m_mXmlParser = new ModelXmlParser(this);
    m_mXmlParser->SetTreeWidget1(m_vtXmlTab->GettwXmlViewerLeft());
    m_mXmlParser->SetTreeWidget2(m_vtXmlTab->GettwXmlViewerRight());
    m_mXmlParser->SetWorkerMode(1);
    m_mXmlParser->SetFileName("test.xml"); // Set the XML file name to be parsed
    m_mXmlParser->SetFileName2("test2.xml"); // Set the second XML file name to be parsed
    connect(m_mXmlParser, SIGNAL(AllPageReaded(QTreeWidget*)), this, SLOT(handleAllPageReaded(QTreeWidget*)));
    m_mXmlParser->start(); // Start parsing XML file in a separate thread

    

    QVBoxLayout *mainLayout = new QVBoxLayout(this);
    mainLayout->setContentsMargins(0, 0, 0, 0);
    mainLayout->addWidget(m_vtXmlTab);

    this->setLayout(mainLayout);

    // this->showMaximized();
}

void LcXml::handleAllPageReaded(QTreeWidget* twTarget)
{
    cout << "All pages have been read" << endl;
    QTreeWidget *tw = twTarget;
    // set column 0 width
    tw->setColumnWidth(0, 200);
    // set column 1 width
    tw->setColumnWidth(1, 250);
    // set column 2 width
    tw->setColumnWidth(2, 100);
    // set column 3 width
    tw->setColumnWidth(3, 100);
    // expand all items in the tree view
    // this will ensure that all nodes are visible when the widget is first shown or re-shown after being minimized and maximized
    tw->setUniformRowHeights(true);
    tw->expandAll();
}
