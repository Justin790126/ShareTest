#ifndef VIEW_CHART_PROPS_H
#define VIEW_CHART_PROPS_H

#include "PropsSection.h"
#include <QtGui>

class ViewChartProps : public PropsSection
{
    Q_OBJECT
public:
    ViewChartProps(const QString& title = "", const int animationDuration = PropsSection::DEFAULT_DURATION, QWidget* parent = nullptr);

    ~ViewChartProps();

signals:
    void chartTitleChanged(const QString& title);
private:
    QLineEdit* leChartTitle;
};


#endif /* VIEW_CHART_PROPS_H */