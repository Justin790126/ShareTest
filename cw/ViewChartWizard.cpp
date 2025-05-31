#include "ViewChartWizard.h"

ViewChartWizard::ViewChartWizard(QWidget* parent)
    : QWidget(parent)
{
    UI();

    resize(800, 400);
}
ViewChartWizard::~ViewChartWizard()
{
    // Destructor implementation
}
void ViewChartWizard::Widgets()
{
    // Create widgets for the chart wizard
    // This method will create all necessary widgets for the wizard
}
void ViewChartWizard::Layouts()
{
    QSplitter *splt = new QSplitter(Qt::Horizontal, this);

  vlytLeft = new QVBoxLayout;
  vlytLeft->setContentsMargins(0, 0, 0, 0);
  {
    vlytLeftProps = new QVBoxLayout;
    vlytLeft->addLayout(vlytLeftProps);
    vlytLeft->addStretch(5);
  }

  QWidget *widLeft = new QWidget(this);
  widLeft->setLayout(vlytLeft);

  QVBoxLayout *vlytRight = new QVBoxLayout;
  vlytRight->setContentsMargins(0, 0, 0, 0);
  {
    m_qcp = new QCustomPlot(this);
    vlytRight->addWidget(m_qcp);
  }
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
void ViewChartWizard::UI()
{
    Widgets();
    Layouts();
}