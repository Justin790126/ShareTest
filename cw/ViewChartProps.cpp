#include "ViewChartProps.h"

ViewChartProps::ViewChartProps(const QString& title, const int animationDuration, QWidget* parent)
    : PropsSection(title, animationDuration, parent)
{
    UI();
}

void ViewChartProps::UI()
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
        QHBoxLayout* hlytXLabel = new QHBoxLayout;
        {
            QLabel* lblXLabel = new QLabel(tr("X Axis Label:"));
            hlytXLabel->addWidget(lblXLabel);
            leXLabel = new QLineEdit(this);
            leXLabel->setPlaceholderText(tr("Enter X axis label here..."));
            hlytXLabel->addWidget(leXLabel);
        }
        QHBoxLayout* hlytYLabel = new QHBoxLayout;
        {
            QLabel* lblYLabel = new QLabel(tr("Y Axis Label:"));
            hlytYLabel->addWidget(lblYLabel);
            leYLabel = new QLineEdit(this);
            leYLabel->setPlaceholderText(tr("Enter Y axis label here..."));
            hlytYLabel->addWidget(leYLabel);
        }
        QHBoxLayout* hlytLegend = new QHBoxLayout;
        {
            chbLegend = new QCheckBox(tr("Show Legend"));
            chbLegend->setChecked(true); // Default to checked
            hlytLegend->addWidget(chbLegend);
        }

        vlyt->addLayout(hlytTitle);
        vlyt->addLayout(hlytXLabel);
        vlyt->addLayout(hlytYLabel);
        vlyt->addLayout(hlytLegend);

        connect(leChartTitle, SIGNAL(textChanged(const QString&)),
                this, SIGNAL(chartTitleChanged(const QString&)));
        connect(leXLabel, SIGNAL(textChanged(const QString&)),
                this, SIGNAL(xLabelChanged(const QString&)));
        connect(leYLabel, SIGNAL(textChanged(const QString&)),
                this, SIGNAL(yLabelChanged(const QString&)));
        connect(chbLegend, SIGNAL(toggled(bool)), 
                this, SIGNAL(legendVisibilityChanged(bool)));
    }
    this->setContentLayout(*vlyt);
}

ViewChartProps::~ViewChartProps()
{
    // Destructor implementation
}