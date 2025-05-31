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

    void UI();

signals:
    void chartTitleChanged(const QString& title);
    void xLabelChanged(const QString& label);
    void yLabelChanged(const QString& label);
    void legendVisibilityChanged(bool visible);
private:
    QLineEdit* leChartTitle;
    QLineEdit* leXLabel;
    QLineEdit* leYLabel;
    QCheckBox* chbLegend;
};


#endif /* VIEW_CHART_PROPS_H */