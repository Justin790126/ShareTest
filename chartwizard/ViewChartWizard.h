#ifndef VIEW_CHART_WIZARD_H
#define VIEW_CHART_WIZARD_H

#include <QWidget>
#include "qcustomplot.h"
#include "Section.h"

enum ToolBoxType
{
    TOOLBOX_CHART_2D_XY,
};

// Create a ChartTypeDialog inherit from QDialog


class ChartTypeDialog : public QDialog
{
    Q_OBJECT
    public:
    ChartTypeDialog(QWidget *parent = 0);
    ~ChartTypeDialog() = default;
    
};


class PropsSection : public Section
{
    Q_OBJECT
public:
    PropsSection(const QString &title = "", const int animationDuration = Section::DEFAULT_DURATION, QWidget *parent = 0, int idx = 0);
    ~PropsSection() = default;


private:
    void Widgets();
    void Layout();

    int m_iSecIdx;
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
    QPushButton* GetNewChartButton() const { return btnNewChart; }
    void AddNewChart(PropsSection* newChart);
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