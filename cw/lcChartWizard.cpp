#include "lcChartWizard.h"

lcChartWizard::lcChartWizard(QWidget *parent) : QWidget(parent) {
  UI();

  resize(800, 400);

  vector<double> x_nms(SLICE_DATA.size());
  double nmPerStep = 4;
  for (int i = 0; i < SLICE_DATA.size(); ++i) {
    x_nms[i] = i * nmPerStep; // Assuming x-axis is just the index
  }
  vector<double> y_noise(SLICE_DATA.size());
  for (int i = 0; i < SLICE_DATA.size(); ++i) {
    // add random noise
    y_noise[i] =
        SLICE_DATA[i] + ((rand() % 100) / 10000.0); // Adding small random noise
  }

  m_qcp->setInteractions(QCP::iSelectPlottables | QCP::iSelectAxes |
                         QCP::iRangeDrag | QCP::iRangeZoom);
  m_qcp->xAxis->setLabel("X Axis (nm)");
  m_qcp->yAxis->setLabel("Y Axis (Intensity)");

  m_qcp->legend->setVisible(true);

  ChartInfo *info = new ChartInfo();
  info->SetX(QVector<double>::fromStdVector(x_nms));
  info->SetY(QVector<double>::fromStdVector(y_noise));
  info->SetPen(QPen(Qt::blue, 2)); // Set line color to blue and width to 2

  info->SetLegendName("Line Chart noises");
  info->SetQCustomPlot(m_qcp);

  QWidget *lineProps = CreateLineChartProps(info);
  m_vWidChartInfo.push_back(make_pair(lineProps, info));
  vlytLeftProps->addWidget(lineProps);

  // add another curve without noise
  ChartInfo *info2 = new ChartInfo();
  info2->SetX(QVector<double>::fromStdVector(x_nms));
  info2->SetY(QVector<double>::fromStdVector(SLICE_DATA));
  info2->SetPen(QPen(Qt::red, 2)); // Set line color to red and width to 2

  info2->SetLegendName("Line Chart without noise");
  info2->SetQCustomPlot(m_qcp);
  QWidget *lineProps2 = CreateLineChartProps(info2);
  m_vWidChartInfo.push_back(make_pair(lineProps2, info2));
  vlytLeftProps->addWidget(lineProps2);

  m_qcp->replot();

  // change graph selection color

  // signal / slot to handle user select the line chart
  connect(m_qcp, SIGNAL(selectionChangedByUser()), this,
          SLOT(handleLineChartSelection()));

  ConnectLineChartProps();
}

void lcChartWizard::ConnectLineChartProps() {
  // Connect signals from ViewLineChartProps to lcChartWizard
  for (const auto &pair : m_vWidChartInfo) {
    ViewLineChartProps *lineProps =
        qobject_cast<ViewLineChartProps *>(pair.first);
    if (lineProps) {
      connect(lineProps, SIGNAL(lineNameChanged(const QString &)), this,
              SLOT(handleLineNameChanged(const QString &)));
    }
  }
}

void lcChartWizard::handleLineNameChanged(const QString &name) {
  // Handle the line name change here
  QString tgtName = "";
  if (name.isEmpty()) {
    tgtName = "";
  } else {
    tgtName = name;
  }

  ViewLineChartProps *lps = qobject_cast<ViewLineChartProps *>(sender());
  if (lps) {
    lps->setTitle(tgtName); // Update the title of the ViewLineChartProps
    // Find the corresponding ChartInfo based on the sender
    for (const auto &pair : m_vWidChartInfo) {
      if (pair.first == lps) {
        ChartInfo *chartInfo = pair.second;
        if (chartInfo) {
          chartInfo->SetLegendName(tgtName);
          // Update the legend name in the QCustomPlot
          int graphIndex = chartInfo->GetGraphIndex();
          m_qcp->graph(graphIndex)->setName(tgtName);
          m_qcp->replot(); // Replot to reflect changes
        }
        break; // Exit loop after finding the first match
      }
    }
  }
}

void lcChartWizard::handleLineChartSelection() {
  QCustomPlot *qcp = (QCustomPlot *)sender();
  // Decide which line chart is selected
  if (qcp->selectedGraphs().isEmpty()) {
    return; // No graph selected
  }
  QCPGraph *selectedGraph = qcp->selectedGraphs().first();

  // find Props based on selectedGraph and m_vWidChartInfo
  QWidget *tgtWidget = NULL;
  for (const auto &pair : m_vWidChartInfo) {
    ChartInfo *chartInfo = pair.second;
    int graphIdx = chartInfo->GetGraphIndex();
    if (qcp->graph(graphIdx) == selectedGraph) {
      // Found the matching chart info
      // You can now use chartInfo to update the UI or perform actions
      qDebug() << "Selected graph index:" << graphIdx;
      qDebug() << "Legend name:" << chartInfo->GetLegendName();
      tgtWidget = pair.first; // Get the corresponding widget
      // Add more actions as needed
      break; // Exit loop after finding the first match
    }
  }

  // convert tgtWidget to ViewLineChartProps and expand the UI, and fold others
  if (tgtWidget) {
    ViewLineChartProps *lineProps =
        qobject_cast<ViewLineChartProps *>(tgtWidget);
    if (lineProps) {
      // Expand the selected line chart properties
      lineProps->setExpanded(true);
      // Optionally, fold other properties
      for (const auto &pair : m_vWidChartInfo) {
        if (pair.first != tgtWidget) {
          ViewLineChartProps *otherProps =
              qobject_cast<ViewLineChartProps *>(pair.first);
          if (otherProps) {
            otherProps->setExpanded(false);
          }
        }
      }
    }
  }
}

lcChartWizard::~lcChartWizard() {
  // Destructor implementation
}

void lcChartWizard::Widgets() {
  // Initialize widgets here
}

QWidget *lcChartWizard::CreateLineChartProps(ChartInfo *chartInfo) {
  ViewLineChartProps *section = new ViewLineChartProps(
      chartInfo ? chartInfo->GetLegendName() : "Default Title");

  // Use chartInfo to draw line chart
  if (chartInfo) {

    chartInfo->SetQCustomPlot(m_qcp);
    // use graphcount to calculate graph index and set

    m_qcp->addGraph();
    int graphCount = m_qcp->graphCount();
    int graphIndx = graphCount - 1; // Use the last graph index
    chartInfo->SetGraphIndex(graphIndx);
    m_qcp->graph(graphIndx)->setData(*chartInfo->GetX(), *chartInfo->GetY());
    // set pen
    m_qcp->graph(graphIndx)->setPen(chartInfo->GetPen());
    // set brush
    m_qcp->graph(graphIndx)->setBrush(chartInfo->GetBrush());
    m_qcp->xAxis->setRange(0, chartInfo->GetX()->back());
    m_qcp->yAxis->setRange(
        *std::min_element(chartInfo->GetY()->begin(), chartInfo->GetY()->end()),
        *std::max_element(chartInfo->GetY()->begin(),
                          chartInfo->GetY()->end()));
    // add scatter style to fill circle
    m_qcp->graph(graphIndx)->setScatterStyle(
        QCPScatterStyle(QCPScatterStyle::ssCircle, 5));
    // set legend name
    m_qcp->graph(graphIndx)->setName(chartInfo->GetLegendName());

    QPen graphPen = chartInfo->GetPen();
    QColor graphColor = graphPen.color();
    QColor complementaryColor(255 - graphColor.red(), 255 - graphColor.green(),
                              255 - graphColor.blue());
    QCPSelectionDecorator *decorator = new QCPSelectionDecorator();
    decorator->setPen(
        QPen(complementaryColor, 2)); // Set selection color to green
    m_qcp->graph(graphIndx)->setSelectionDecorator(decorator);
  }
  return section;
}

void lcChartWizard::Layouts() {
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

void lcChartWizard::UI() {
  Widgets();
  Layouts();
}