
#ifndef CHART_WIZARD_H
#define CHART_WIZARD_H

#include <QWidget>
#include <QTabWidget>
#include <QVBoxLayout>
#include <QGroupBox>
#include "ViewChartWizard.h"
// Controller classes

class ChartWizard 
{
    public:
        ChartWizard();
        ~ChartWizard() = default;
    private:
        ViewChartWizard * view;
};

#endif // CHART_WIZARD_H