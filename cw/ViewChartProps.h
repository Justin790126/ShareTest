#ifndef VIEW_CHART_PROPS_H
#define VIEW_CHART_PROPS_H

#include "PropsSection.h"

class ViewChartProps : public PropsSection
{
    Q_OBJECT
public:
    ViewChartProps(const QString& title = "", const int animationDuration = PropsSection::DEFAULT_DURATION, QWidget* parent = nullptr);

    ~ViewChartProps();

private:
    
};


#endif /* VIEW_CHART_PROPS_H */