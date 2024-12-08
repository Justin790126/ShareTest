#ifndef VIEW_CHART_WIZARD_H
#define VIEW_CHART_WIZARD_H

#include <QWidget>
#include "qcustomplot.h"
#include "Section.h"
#include "utils.h"

// Create a ChartTypeDialog inherit from QDialog

class ChartTypeDialog : public QDialog
{
    Q_OBJECT
public:
    ChartTypeDialog(QWidget *parent = 0);
    ~ChartTypeDialog() = default;

    void Widgets();
    void Layout();
    void Connect();

    int GetChartTypeIdx() const { return m_iChartTypeIdx; }

    QButtonGroup *btngChartType;

private:
    QPushButton *btnOk;
    QPushButton *btnCancel;

    int m_iChartTypeIdx;

private slots:
    void handleBtnOkClicked();
    void handleBtnCancelClicked();
    void handleChartTypeChanged(int chartType);
};

class PropsSection : public Section
{
    Q_OBJECT
public:
    PropsSection(const QString &title = "", const int animationDuration = Section::DEFAULT_DURATION, QWidget *parent = 0, int idx = 0);
    ~PropsSection() = default;
    void SetQcp(QCustomPlot *plot) { m_pQcp = plot; }
    QCustomPlot *GetQcp() const { return m_pQcp; }
    void SetGraph(QCPGraph *graph) { m_pGraph = graph; }
    QCPGraph *GetGraph() const { return m_pGraph; }

    // set m_xData without memory copy
    void SetXData(const QVector<double> &data);
    void SetXLabel(const QString &label);
    void SetYData(const QVector<double> &data);
    void SetYLabel(const QString &label);

    void Plot();

    QComboBox *cbbX;
    QLineEdit *leXaxis;
    QComboBox *cbbY;
    QLineEdit *leYaxis;

    QComboBox *cbbZ;
    QLineEdit *leZaxis;
signals:
    void xAxisChanged(int idx);
    void yAxisChanged(int idx);
    void zAxisChanged(int idx);

    void xAxisTextChanged(const QString &);
    void yAxisTextChanged(const QString &);

private:
    void Widgets();
    void Layout();
    void Connect();

    QVector<double> m_xData;
    QVector<double> m_yData;
    QVector<double> m_zData;

    int m_iSecIdx;

    QCustomPlot *m_pQcp;
    QCPGraph *m_pGraph;
private slots:
    void handleXAxisChanged(int idx);
    void handleYAxisChanged(int idx);
    void handleXaxisTextChanged(const QString &);
    void handleYaxisTextChanged(const QString &);
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
    QPushButton *GetNewChartButton() const { return btnNewChart; }
    void AddNewChart(PropsSection *newChart);

    QCustomPlot *GetQcp() const { return qcp; }

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