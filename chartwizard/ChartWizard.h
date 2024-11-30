
#ifndef CHART_WIZARD_H
#define CHART_WIZARD_H

#include <QWidget>
#include <QTabWidget>
#include <QVBoxLayout>
#include <QGroupBox>
#include "ViewChartWizard.h"
// Controller classes

class ChartWizard : public QWidget
{
    Q_OBJECT
    public:
        ChartWizard(QWidget *parent=NULL);
        ~ChartWizard() = default;
    private:
        ViewChartWizard * view;
};

#endif // CHART_WIZARD_H