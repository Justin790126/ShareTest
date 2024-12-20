#include "lcXml.h"

LcXml::LcXml(QWidget *parent) : QWidget(parent)
{
    // Add UI elements here
    resize(640, 480);

    m_mXmlParser = new ModelXmlParser(this);
    m_mXmlParser->SetFileName("test.xml"); // Set the XML file name to be parsed
    connect(m_mXmlParser, SIGNAL(AllPageReaded()), this, SLOT(handleAllPageReaded()));
    m_mXmlParser->start(); // Start parsing XML file in a separate thread



    m_vtXmlTab = new ViewXmlTab(this);

    QVBoxLayout *mainLayout = new QVBoxLayout(this);
    mainLayout->setContentsMargins(0,0,0,0);
    mainLayout->addWidget(m_vtXmlTab);

    this->setLayout(mainLayout);
}


void LcXml::handleAllPageReaded()
{
    cout<< "All pages have been read" << endl;
}
