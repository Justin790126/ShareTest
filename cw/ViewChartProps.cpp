#include "ViewChartProps.h"

ViewChartProps::ViewChartProps(const QString& title, const int animationDuration, QWidget* parent)
    : PropsSection(title, animationDuration, parent)
{
    QVBoxLayout *vlyt = new QVBoxLayout;
    {
        QHBoxLayout* hlytTitle = new QHBoxLayout;
        {
            QLabel* lblTitle = new QLabel(tr("Chart Title:"));
            hlytTitle->addWidget(lblTitle);
            leChartTitle = new QLineEdit(this);
            leChartTitle->setPlaceholderText(tr("Enter chart title here..."));
            hlytTitle->addWidget(leChartTitle);
        }

        vlyt->addLayout(hlytTitle);

        connect(leChartTitle, SIGNAL(textChanged(const QString&)),
                this, SIGNAL(chartTitleChanged(const QString&)));
    }
    this->setContentLayout(*vlyt);
}
ViewChartProps::~ViewChartProps()
{
    // Destructor implementation
}