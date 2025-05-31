#ifndef VIEW_LINE_CHART_PROPS_H
#define VIEW_LINE_CHART_PROPS_H

#include "ViewChartProps.h"

#include <QtGui>

class ViewLineChartProps : public ViewChartProps
{
    Q_OBJECT
public:
    ViewLineChartProps(const QString& title = "", const int animationDuration = PropsSection::DEFAULT_DURATION, QWidget* parent = nullptr);

    ~ViewLineChartProps();

signals:
    void lineNameChanged(const QString& name);

private:
    QLineEdit* leLineName;

};

#endif /* VIEW_LINE_CHART_PROPS_H */