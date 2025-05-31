#ifndef VIEW_LINE_CHART_PROPS_H
#define VIEW_LINE_CHART_PROPS_H

#include "ViewChartProps.h"
#include "qcustomplot.h"
#include <QtGui>

class ViewLineChartProps : public ViewChartProps
{
    Q_OBJECT
public:
    ViewLineChartProps(const QString& title = "", const int animationDuration = PropsSection::DEFAULT_DURATION, QWidget* parent = nullptr);

    ~ViewLineChartProps();

    QComboBox* getDotStyleComboBox() const { return cbbDotStyle; }

signals:
    void lineNameChanged(const QString& name);
    void dotStyleChanged(int index);

private:
    QLineEdit* leLineName;
    QComboBox* cbbDotStyle;

};

#endif /* VIEW_LINE_CHART_PROPS_H */