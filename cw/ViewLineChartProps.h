#ifndef VIEW_LINE_CHART_PROPS_H
#define VIEW_LINE_CHART_PROPS_H

#include "ViewChartProps.h"
#include "qcustomplot.h"
#include <QtGui>

class ViewLineChartProps : public PropsSection
{
    Q_OBJECT
public:
    ViewLineChartProps(const QString& title = "", const int animationDuration = PropsSection::DEFAULT_DURATION, QWidget* parent = nullptr);

    ~ViewLineChartProps();

    QComboBox* getDotStyleComboBox() const { return cbbDotStyle; }
    QDoubleSpinBox* getDotSizeSpinBox() const { return dsbDotSize; }
    QDoubleSpinBox* getLineWidthSpinBox() const { return dsbLineWidth; }

    void UI();

signals:
    void lineNameChanged(const QString& name);
    void dotStyleChanged(int index);
    void dotSizeChanged(double size);
    void showLineSegmentChanged(bool checked);
    void lineWidthChanged(double width);
    void lineColorChanged(const QString& color);
    void showGraphChanged(bool checked);
private:
    QLineEdit* leLineName;
    QComboBox* cbbDotStyle;
    QDoubleSpinBox* dsbDotSize;

    QCheckBox* chbShowLineSegment;
    QDoubleSpinBox* dsbLineWidth;

    QLineEdit* leLineColor;
    QCheckBox* chbShowGraph;
};

#endif /* VIEW_LINE_CHART_PROPS_H */