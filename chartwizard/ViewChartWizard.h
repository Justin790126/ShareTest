#ifndef VIEW_CHART_WIZARD_H
#define VIEW_CHART_WIZARD_H

#include <QWidget>
#include "qcustomplot.h"


class ViewChartWizard : public QWidget
{
private:
    /* data */
public:
    ViewChartWizard(QWidget *parent=NULL);
    ~ViewChartWizard();
};



#endif // VIEW_CHART_WIZARD_H