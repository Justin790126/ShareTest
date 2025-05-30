#ifndef VIEW_LINE_CHART_PROPS_H
#define VIEW_LINE_CHART_PROPS_H

#include "ViewChartProps.h"

class ViewLineChartProps : public ViewChartProps
{
    Q_OBJECT
public:
    ViewLineChartProps(const QString& title = "", const int animationDuration = PropsSection::DEFAULT_DURATION, QWidget* parent = nullptr);

    ~ViewLineChartProps();

private:
};

#endif /* VIEW_LINE_CHART_PROPS_H */