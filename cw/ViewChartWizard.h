#ifndef VIEW_CHART_WIZARD_H
#define VIEW_CHART_WIZARD_H

#include "ViewLineChartProps.h"
#include "qcustomplot.h"
#include <QtGui>

class ViewChartWizard : public QWidget {
  Q_OBJECT
public:
  ViewChartWizard(QWidget *parent = nullptr);
  ~ViewChartWizard();
  void Widgets();
  void Layouts();
  void UI();

  QVBoxLayout *getVLayoutLeftProps() const { return vlytLeftProps; }
  QCustomPlot *getQCustomPlot() const { return m_qcp; }
  void setQCustomPlot(QCustomPlot *qcp) { m_qcp = qcp; }

private:
  QCustomPlot *m_qcp;
  QVBoxLayout *vlytLeftProps;
  QVBoxLayout *vlytLeft;
};

#endif /* VIEW_CHART_WIZARD_H */