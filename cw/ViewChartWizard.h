#ifndef VIEW_CHART_WIZARD_H
#define VIEW_CHART_WIZARD_H

#include "ViewChartProps.h"
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
  ViewChartProps *getVCPGeneral() const { return vcpGeneral; }
  QCPTextElement *getTitleElement() const { return m_pTitleElement; }

private:
  QCustomPlot *m_qcp;
  QVBoxLayout *vlytLeftProps;
  QVBoxLayout *vlytLeft;

  ViewChartProps* vcpGeneral;
  QCPTextElement *m_pTitleElement;
};

#endif /* VIEW_CHART_WIZARD_H */