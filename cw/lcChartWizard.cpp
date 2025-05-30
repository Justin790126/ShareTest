#include "lcChartWizard.h"

#include "data.h"

lcChartWizard::lcChartWizard(QWidget* parent)
    : QWidget(parent)
{
    UI();

    resize(640, 480);


}

lcChartWizard::~lcChartWizard()
{
    // Destructor implementation
}

void lcChartWizard::Widgets()
{
    // Initialize widgets here
    

}

QWidget* lcChartWizard::CreateLineChartProps()
{
    ViewLineChartProps* section = new ViewLineChartProps("Line Chart Properties");
    {
        QVBoxLayout* vlyt = new QVBoxLayout;
        
        vlyt->addWidget(new QLabel("Line Chart Properties", this));

        section->setContentLayout(*vlyt);
    }
    return section;
}

void lcChartWizard::Layouts()
{
    QSplitter* splt = new QSplitter(Qt::Horizontal, this);

    QVBoxLayout* vlytLeft = new QVBoxLayout;
    vlytLeft->setContentsMargins(0, 0, 0, 0);
    {
        vlytLeft->addStretch(1); // Add stretch to fill remaining space
    }

    QWidget* widLeft = new QWidget(this);
    widLeft->setLayout(vlytLeft);


    QVBoxLayout* vlytRight = new QVBoxLayout;
    vlytRight->setContentsMargins(0, 0, 0, 0);
    {
        m_qcp = new QCustomPlot(this);
        vlytRight->addWidget(m_qcp);
    }
    QWidget* widRight = new QWidget(this);
    widRight->setLayout(vlytRight);
    
    splt->addWidget(widLeft);
    splt->addWidget(widRight);
    splt->setSizes({200, 400}); // Set initial sizes for the splitter

    QVBoxLayout* vlytMain = new QVBoxLayout(this);
    vlytMain->setContentsMargins(0, 0, 0, 0);
    vlytMain->addWidget(splt);
    setLayout(vlytMain);
}

void lcChartWizard::UI()
{
    Widgets();
    Layouts();
}