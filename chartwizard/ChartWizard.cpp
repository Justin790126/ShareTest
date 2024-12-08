
#include "ChartWizard.h"

QVector<double> x(101), y(101);

ChartWizard::ChartWizard()
{
    for (int i = 0; i < 101; ++i)
    {
        x[i] = i / 100.0;             // x goes from 0 to 1
        y[i] = qSin(x[i] * 2 * M_PI); // let's plot a sine curve
    }
    view = new ViewChartWizard();
    view->show();

    QPushButton *btnNewChart = view->GetNewChartButton();
    connect(btnNewChart, SIGNAL(clicked()), this, SLOT(handleNewChartCreated()));
}

void ChartWizard::handleXaxisChanged(int idx)
{
    PropsSection *secChart = static_cast<PropsSection *>(sender());
    if (secChart)
    {

        if (idx == 0)
        {

            secChart->SetXData(x);
            secChart->Plot();
        }
        else if (idx == 1)
        {

            secChart->SetXData(y);
            secChart->Plot();
            // secChart->GetGraph()->setData(x, y);
            // secChart->GetQcp()->replot();
            // secChart->GetQcp()->clearGraphs();
            // secChart->GetQcp()->replot();
        }
        else
        {
        }
        cout << "X-axis changed to index: " << idx << endl;
    }
}

void ChartWizard::handleYaxisChanged(int idx)
{

    PropsSection *secChart = static_cast<PropsSection *>(sender());
    if (secChart)
    {
        cout << "Y-axis changed to index: " << idx << endl;
        if (idx == 0)
        {

            secChart->SetYData(x);
            secChart->Plot();
        }
        else if (idx == 1)
        {

            secChart->SetYData(y);
            secChart->Plot();
        }
    }
}

void ChartWizard::handleNewChartCreated()
{
    if (!m_chartTypeDialog)
        m_chartTypeDialog = new ChartTypeDialog(view);
    if (m_chartTypeDialog->exec() == QDialog::Accepted)
    {
        int chartTypeIdx = m_chartTypeDialog->GetChartTypeIdx();
        if (chartTypeIdx == -1)
            return;

        cout << "New chart created: " << chartTypeIdx << endl;

        PropsSection *secChart = new PropsSection("Chart", 0, NULL, chartTypeIdx);
        connect(secChart, SIGNAL(sectionClosed()), this, SLOT(handleSectionClosed()));
        QCustomPlot *plt = view->GetQcp();
        QCPGraph *chart = plt->addGraph();
        secChart->SetQcp(plt);
        secChart->SetGraph(chart);

        if (chartTypeIdx == 0)
        {
            connect(secChart, SIGNAL(xAxisChanged(int)), this, SLOT(handleXaxisChanged(int)));
            connect(secChart, SIGNAL(yAxisChanged(int)), this, SLOT(handleYaxisChanged(int)));
            connect(secChart, SIGNAL(xAxisTextChanged(const QString &)), this, SLOT(handleXaxisTextChanged(const QString &)));
            connect(secChart, SIGNAL(yAxisTextChanged(const QString &)), this, SLOT(handleYaxisTextChanged(const QString &)));
        }

        m_vPropsSections.emplace_back(secChart);
        view->AddNewChart(secChart);
    }
    if (m_chartTypeDialog)
        delete m_chartTypeDialog;
    m_chartTypeDialog = NULL;
}

void ChartWizard::handleXaxisTextChanged(const QString &lbl)
{
    PropsSection *secChart = static_cast<PropsSection *>(sender());
    if (secChart)
    {
        secChart->SetXLabel(lbl);
    }
}
void ChartWizard::handleYaxisTextChanged(const QString &lbl)
{
    PropsSection *secChart = static_cast<PropsSection *>(sender());
    if (secChart)
    {
        secChart->SetYLabel(lbl);
    }
}

void ChartWizard::handleSectionClosed()
{
    PropsSection *closedSection = static_cast<PropsSection *>(sender());
    auto it = find(m_vPropsSections.begin(), m_vPropsSections.end(), closedSection);
    if (it != m_vPropsSections.end())
    {
        m_vPropsSections.erase(it);
    }
    delete closedSection;
}