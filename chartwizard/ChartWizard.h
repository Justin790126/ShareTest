
#ifndef CHART_WIZARD_H
#define CHART_WIZARD_H

#include <QWidget>
#include <QTabWidget>
#include <QVBoxLayout>
#include <QGroupBox>
#include "ViewChartWizard.h"

#include <iostream>
#include <vector>

using namespace std;
// Controller classes

class ChartWizard : public QObject
{
    Q_OBJECT
public:
    ChartWizard();
    ~ChartWizard() = default;

private:
    ViewChartWizard *view;
    vector<PropsSection *> m_vPropsSections;
    ChartTypeDialog *m_chartTypeDialog;

private slots:
    void handleNewChartCreated();
    void handleSectionClosed();
};

#endif // CHART_WIZARD_H