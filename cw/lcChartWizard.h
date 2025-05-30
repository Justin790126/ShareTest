#ifndef LC_CHARTWIZARD_H
#define LC_CHARTWIZARD_H

#include <QtGui>
#include "ViewLineChartProps.h"
#include "qcustomplot.h"

class lcChartWizard : public QWidget
{
    Q_OBJECT
public:
    lcChartWizard(QWidget* parent = nullptr);
    ~lcChartWizard();
    QWidget* CreateLineChartProps();
    QVBoxLayout* vlytLeft;
private:
    void Widgets();
    void Layouts();
    void UI();

    

    QCustomPlot* m_qcp;
};


#endif /* LC_CHARTWIZARD_H */