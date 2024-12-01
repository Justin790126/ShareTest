#ifndef VIEW_CHART_WIZARD_H
#define VIEW_CHART_WIZARD_H

#include <QWidget>
#include "qcustomplot.h"
#include "Section.h"

enum ToolBoxType
{
    TOOLBOX_CHART_2D_XY,
};

class PropsSection : public Section
{
    Q_OBJECT
public:
    PropsSection(const QString &title = "", const int animationDuration = Section::DEFAULT_DURATION, QWidget *parent = 0);
    ~PropsSection() = default;

private:
    void Widgets();
    void Layout();
};

class ViewChartWizard : public QWidget
{
    Q_OBJECT
public:
    ViewChartWizard(QWidget *parent = NULL);
    ~ViewChartWizard() = default;

    void Widgets();
    void Layout();
    void CreateTableWidget();
    void CreatePropWidget();

private:
    QCustomPlot *qcp;
    QWidget *widProps;
    QWidget *widTable;
    QWidget *widChart;

    QTableWidget *tbwCSVcontent;
    QPushButton *btnNewChart;
    QVBoxLayout *vlytProps;
    QScrollArea *sclProps;

private slots:
    void handleNewChartCreated();
};

#endif // VIEW_CHART_WIZARD_H