#include "ViewGpuSetup.h"

ViewGpuSetup::ViewGpuSetup(QWidget *parent) :
    QWidget(parent)
{
    setupUi();
}

ViewGpuSetup::~ViewGpuSetup()
{
}

void ViewGpuSetup::setupUi()
{
    // Create main horizontal layout
    mainLayout = new QHBoxLayout(this);
    
    // Create left side - QTreeWidget
    treeWidget = new QTreeWidget(this);
    treeWidget->setHeaderHidden(true);
    treeWidget->setMinimumWidth(200);
    treeWidget->setMaximumWidth(300);
    
    // Create right side - QStackedWidget
    stackedWidget = new QStackedWidget(this);
    
    // Add widgets to layout
    mainLayout->addWidget(treeWidget);
    mainLayout->addWidget(stackedWidget);
    
    // Set stretch factors (treeWidget fixed, stackedWidget expands)
    mainLayout->setStretchFactor(treeWidget, 0);
    mainLayout->setStretchFactor(stackedWidget, 1);
    
    // Set layout to widget
    setLayout(mainLayout);
}