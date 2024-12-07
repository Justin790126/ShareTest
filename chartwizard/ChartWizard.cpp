
#include "ChartWizard.h"

ChartWizard::ChartWizard() 
{
   
    view = new ViewChartWizard();
    view->show();

    QPushButton* btnNewChart = view->GetNewChartButton();
    connect(btnNewChart, SIGNAL(clicked()), this, SLOT(handleNewChartCreated()));
}

void ChartWizard::handleNewChartCreated()
{
    ChartTypeDialog* dialog = new ChartTypeDialog(view);
    if (dialog->exec() == QDialog::Accepted) {
        // QString chartType = dialog->GetSelectedChartType();
        // Create and add the chart to the view
        PropsSection *secChart = new PropsSection("Chart", 0);
    connect(secChart, SIGNAL(sectionClosed()), this, SLOT(handleSectionClosed()));

    m_vPropsSections.emplace_back(secChart);
    view->AddNewChart(secChart);
    }
    
}

void ChartWizard::handleSectionClosed()
{
    PropsSection* closedSection = static_cast<PropsSection*>(sender());
    auto it = find(m_vPropsSections.begin(), m_vPropsSections.end(), closedSection);
    if (it!=m_vPropsSections.end()) {
        m_vPropsSections.erase(it);
    }
    delete closedSection;
}