#include "lcXml.h"

LcXml::LcXml(QWidget *parent) : QWidget(parent)
{
    m_vtXmlTab = new ViewXmlTab(this);
    // Add UI elements here
    resize(640, 480);

    m_mXmlParser = new ModelXmlParser(this);
    m_mXmlParser->SetTreeWidget(m_vtXmlTab->GettwXmlViewerLeft());
    m_mXmlParser->SetWorkerMode(1);
    m_mXmlParser->SetFileName("test.xml"); // Set the XML file name to be parsed
    m_mXmlParser->SetFileName2("test2.xml"); // Set the second XML file name to be parsed
    connect(m_mXmlParser, SIGNAL(AllPageReaded()), this, SLOT(handleAllPageReaded()));
    m_mXmlParser->start(); // Start parsing XML file in a separate thread

    

    QVBoxLayout *mainLayout = new QVBoxLayout(this);
    mainLayout->setContentsMargins(0, 0, 0, 0);
    mainLayout->addWidget(m_vtXmlTab);

    this->setLayout(mainLayout);

    // this->showMaximized();
}

void LcXml::handleAllPageReaded()
{
    cout << "All pages have been read" << endl;
    QTreeWidget *tw = m_vtXmlTab->GettwXmlViewerLeft();

    // // add top level
    // QTreeWidgetItem *root = new QTreeWidgetItem(tw);
    // root->setText(0, "Root");
    // // add children
    // QTreeWidgetItem *child1 = new QTreeWidgetItem(root);
    // child1->setText(0, "Child 1");
    // // set text one column 1
    // child1->setText(1, "Text 1");
    // // set text one column 2
    // child1->setText(2, "Attribute 1");

    // // add another child
    // QTreeWidgetItem *child2 = new QTreeWidgetItem(root);
    // child2->setText(0, "Child 2");
    // // set text one column 1
    // child2->setText(1, "Text 2");
    // // set text one column 2
    // child2->setText(2, "Attribute 2");

    // root->addChild(child1);
    // root->addChild(child2);
    // tw->addTopLevelItem(root);
    tw->resizeColumnToContents(0);
    tw->resizeColumnToContents(1);
    tw->resizeColumnToContents(2);
    tw->resizeColumnToContents(3);
    tw->expandAll();
}
