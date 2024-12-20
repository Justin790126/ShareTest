#include "lcXml.h"

LcXml::LcXml(QWidget *parent) : QWidget(parent)
{
    // Add UI elements here
    resize(640, 480);

    m_vtXmlTab = new ViewXmlTab(this);

    QVBoxLayout *mainLayout = new QVBoxLayout(this);
    mainLayout->setContentsMargins(0,0,0,0);
    mainLayout->addWidget(m_vtXmlTab);

    this->setLayout(mainLayout);
}

