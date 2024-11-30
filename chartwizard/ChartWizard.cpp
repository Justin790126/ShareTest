
#include "ChartWizard.h"

ChartWizard::ChartWizard(QWidget *parent) : QWidget(parent)
{
    QVBoxLayout *layout = new QVBoxLayout;
    view = new ViewChartWizard();
    this->setLayout(layout);
}
