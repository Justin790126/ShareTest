
#include "ChartWizard.h"

ChartWizard::ChartWizard()
{

    view = new ViewChartWizard();
    view->show();

    

    QPushButton *btnNewChart = view->GetNewChartButton();
    connect(btnNewChart, SIGNAL(clicked()), this, SLOT(handleNewChartCreated()));
}

void ChartWizard::handleNewChartCreated()
{
    if (!m_chartTypeDialog) m_chartTypeDialog = new ChartTypeDialog(view);
    if (m_chartTypeDialog->exec() == QDialog::Accepted)
    {
        int chartTypeIdx = m_chartTypeDialog->GetChartTypeIdx();
        if (chartTypeIdx == -1) return;

        cout << "New chart created: " << chartTypeIdx << endl;
        // QString chartType = dialog->GetSelectedChartType();
        // Create and add the chart to the view
        PropsSection *secChart = new PropsSection("Chart", 0, NULL, chartTypeIdx);
        connect(secChart, SIGNAL(sectionClosed()), this, SLOT(handleSectionClosed()));

        m_vPropsSections.emplace_back(secChart);
        view->AddNewChart(secChart);
    }
    if (m_chartTypeDialog) delete m_chartTypeDialog;
    m_chartTypeDialog = NULL;
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