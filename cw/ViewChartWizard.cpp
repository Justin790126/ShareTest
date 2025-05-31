#include "ViewChartWizard.h"
#include "ViewColorCombobox.h"

ViewChartWizard::ViewChartWizard(QWidget *parent) : QWidget(parent) {
  UI();
  setWindowTitle(tr("LithoViewer - Chart Wizard"));
  resize(800, 400);
}
ViewChartWizard::~ViewChartWizard() {
  // Destructor implementation
}
void ViewChartWizard::Widgets() {

  m_qcp = new QCustomPlot(this);

  m_qcp->setInteractions(QCP::iSelectPlottables | QCP::iSelectAxes |
                         QCP::iRangeDrag | QCP::iRangeZoom);
  m_qcp->xAxis->setLabel("X Axis (nm)");
  m_qcp->yAxis->setLabel("Y Axis (Intensity)");
  m_qcp->legend->setVisible(true);
  // add chart title
  m_qcp->plotLayout()->insertRow(0);
  m_qcp->plotLayout()->addElement(
      0, 0,
      new QCPTextElement(m_qcp, tr("Chart Title"),
                         QFont("sans", 12, QFont::Bold)));
}
void ViewChartWizard::Layouts() {
  QSplitter *splt = new QSplitter(Qt::Horizontal, this);

  vlytLeft = new QVBoxLayout;
  vlytLeft->setContentsMargins(0, 0, 0, 0);
  {
    vlytLeftProps = new QVBoxLayout;
    vlytLeft->addLayout(vlytLeftProps);
    {
      vcpGeneral = new ViewChartProps(tr("General"), 33, this);
      vlytLeftProps->addWidget(vcpGeneral);
    }

    vlytLeft->addStretch(5);
  }

  QWidget *widLeft = new QWidget(this);
  widLeft->setLayout(vlytLeft);

  QVBoxLayout *vlytRight = new QVBoxLayout;
  vlytRight->setContentsMargins(0, 0, 0, 0);
  { vlytRight->addWidget(m_qcp); }
  QWidget *widRight = new QWidget(this);
  widRight->setLayout(vlytRight);

  splt->addWidget(widLeft);
  splt->addWidget(widRight);
  splt->setSizes({200, 400}); // Set initial sizes for the splitter

  QVBoxLayout *vlytMain = new QVBoxLayout(this);
  vlytMain->setContentsMargins(0, 0, 0, 0);
  vlytMain->addWidget(splt);
  setLayout(vlytMain);
}
void ViewChartWizard::UI() {
  Widgets();
  Layouts();
}