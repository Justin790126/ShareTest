#include "ViewLineChartProps.h"

ViewLineChartProps::ViewLineChartProps(const QString& title, const int animationDuration, QWidget* parent)
    : ViewChartProps(title, animationDuration, parent)
{
    // Constructor implementation
    QVBoxLayout *vlyt = new QVBoxLayout;
    {
        QHBoxLayout* hlytLineName = new QHBoxLayout;
        {
            QLabel* lblLineName = new QLabel(tr("Line Name:"));
            hlytLineName->addWidget(lblLineName);
            leLineName = new QLineEdit;
            leLineName->setPlaceholderText(tr("Enter line name"));
            hlytLineName->addWidget(leLineName);
        }
        vlyt->addLayout(hlytLineName);

        connect(leLineName, SIGNAL(textChanged(const QString&)),
                this, SIGNAL(lineNameChanged(const QString&)));
    }
    this->setContentLayout(*vlyt);
}

ViewLineChartProps::~ViewLineChartProps()
{
    // Destructor implementation
}