#include "ViewChartProps.h"

ViewChartProps::ViewChartProps(const QString& title, const int animationDuration, QWidget* parent)
    : PropsSection(title, animationDuration, parent)
{
    QVBoxLayout *vlyt = new QVBoxLayout;
    {

    }
    this->setContentLayout(*vlyt);
}
ViewChartProps::~ViewChartProps()
{
    // Destructor implementation
}