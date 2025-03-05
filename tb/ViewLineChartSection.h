#ifndef VIEW_LINE_CHART_SECTION_H
#define VIEW_LINE_CHART_SECTION_H

#include <random>
#include <QtGui>
#include "PropsSection.h"
#include "qcustomplot.h"
#include "utils.h"

// write a class inherit PropsSection
class ViewLineChartProps : public PropsSection
{
    Q_OBJECT
public:
    ViewLineChartProps(const QString &title = "", const int animationDuration = DEFAULT_DURATION, QWidget *parent = NULL)
        : PropsSection(title, animationDuration, parent) {}
    ~ViewLineChartProps();
    void SetQcp(QCustomPlot *plot) { m_qcp = plot; }
    QCustomPlot *GetQcp() { return m_qcp; }

    void DrawLineChart(ChartInfo *info);

    void EditChartInfoBy(int idx, ChartInfo* info);
    void UpdateLineChartBy(int idx);

    vector<ChartInfo*>* GetChartInfos() { return &m_vcInfos; }

    void SetLineChartVisibility(int idx, bool visible);


private:
    QCustomPlot *m_qcp = NULL;

    vector<ChartInfo*> m_vcInfos;
};


class CustomPlot : public QCustomPlot {
    Q_OBJECT
public:
    CustomPlot(QWidget *parent = nullptr) : QCustomPlot(parent) {
        // Vertical line
        verticalLine = new QCPItemLine(this);
        verticalLine->setPen(QPen(Qt::black));
        verticalLine->setVisible(false);

        // Horizontal line
        horizontalLine = new QCPItemLine(this);
        horizontalLine->setPen(QPen(Qt::black));
        horizontalLine->setVisible(false);

        // Connect mouse move signal using SIGNAL/SLOT macros
        connect(this, SIGNAL(mouseMove(QMouseEvent*)), this, SLOT(onMouseMove(QMouseEvent*)));

        // Optional: Enable interactions
        setInteractions(QCP::iRangeDrag | QCP::iRangeZoom | QCP::iSelectPlottables);
    }

private slots:
    void onMouseMove(QMouseEvent *event) {
        // Convert pixel position to plot coordinates
        double mouseX = xAxis->pixelToCoord(event->pos().x());
        double mouseY = yAxis->pixelToCoord(event->pos().y());

        // Set vertical line (spans full y-axis range)
        verticalLine->start->setCoords(mouseX, yAxis->range().lower);
        verticalLine->end->setCoords(mouseX, yAxis->range().upper);
        verticalLine->setVisible(true);

        // Set horizontal line (spans full x-axis range)
        horizontalLine->start->setCoords(xAxis->range().lower, mouseY);
        horizontalLine->end->setCoords(xAxis->range().upper, mouseY);
        horizontalLine->setVisible(true);

        // Find and display data values for all graphs at mouseX
        QString tooltipText;
        for (int i = 0; i < graphCount(); ++i) {
            QCPGraph *graph = this->graph(i);
            if (graph) {
                QCPGraphDataContainer::const_iterator it = graph->data()->findBegin(mouseX);
                if (it != graph->data()->end()) {
                    double yValue = it->value;
                    tooltipText += QString("Graph %1: (%2, %3)\n").arg(i).arg(mouseX, 0, 'f', 2).arg(yValue, 0, 'f', 2);
                }
            }
        }

        // Show tooltip at mouse position
        QToolTip::showText(event->globalPos(), tooltipText, this);

        // Redraw the plot
        replot();
    }

private:
    QCPItemLine *verticalLine;
    QCPItemLine *horizontalLine;
};

#endif /* VIEW_LINE_CHART_SECTION_H */