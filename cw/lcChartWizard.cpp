#include "lcChartWizard.h"

lcChartWizard::lcChartWizard(QWidget *parent) {
  // UI();
  vcw = new ViewChartWizard(parent);

  vcw->show();

  QCustomPlot *qcp = vcw->getQCustomPlot();
  QVBoxLayout *vlytLeftProps = vcw->getVLayoutLeftProps();

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



  ModelChartInfo *info = new ModelChartInfo();
  info->SetX(QVector<double>::fromStdVector(x_nms));
  info->SetY(QVector<double>::fromStdVector(y_noise));
  info->SetPen(QPen(Qt::blue, 2)); // Set line color to blue and width to 2

  info->SetLegendName("Line Chart noises");
  info->SetQCustomPlot(qcp);

  QWidget *lineProps = CreateLineChartProps(info);
  m_vWidModelChartInfo.push_back(make_pair(lineProps, info));
  vlytLeftProps->addWidget(lineProps);

  // add another curve without noise
  ModelChartInfo *info2 = new ModelChartInfo();
  info2->SetX(QVector<double>::fromStdVector(x_nms));
  info2->SetY(QVector<double>::fromStdVector(SLICE_DATA));
  info2->SetPen(QPen(Qt::red, 2)); // Set line color to red and width to 2

  info2->SetLegendName("Line Chart without noise");
  info2->SetQCustomPlot(qcp);
  QWidget *lineProps2 = CreateLineChartProps(info2);
  m_vWidModelChartInfo.push_back(make_pair(lineProps2, info2));
  vlytLeftProps->addWidget(lineProps2);

  qcp->replot();

  // change graph selection color

  // signal / slot to handle user select the line chart
  connect(qcp, SIGNAL(selectionChangedByUser()), this,
          SLOT(handleLineChartSelection()));
  ConnectGeneralProps();
  ConnectLineChartProps();
}

void lcChartWizard::ConnectGeneralProps()
{
  ViewChartProps *vcpGeneral = vcw->getVCPGeneral();
  if (vcpGeneral) {
    connect(vcpGeneral, SIGNAL(chartTitleChanged(const QString &)), this,
            SLOT(handleGeneralTitleChanged(const QString &)));
  }
}

void lcChartWizard::handleGeneralTitleChanged(const QString &title) {
  // get title element and settext
  QCustomPlot *qcp = vcw->getQCustomPlot();
  QCPTextElement *titleElement = vcw->getTitleElement();
  if (titleElement) {
    titleElement->setText(title);
    qcp->replot(); // Replot to reflect changes
  } else {
  } 
}

void lcChartWizard::ConnectLineChartProps() {
  // Connect signals from ViewLineChartProps to lcChartWizard
  for (const auto &pair : m_vWidModelChartInfo) {
    ViewLineChartProps *lineProps =
        qobject_cast<ViewLineChartProps *>(pair.first);
    if (lineProps) {
      connect(lineProps, SIGNAL(lineNameChanged(const QString &)), this,
              SLOT(handleLineNameChanged(const QString &)));
      connect(lineProps, SIGNAL(dotStyleChanged(int)), this,
              SLOT(handleDotStyleChanged(int)));
      connect(lineProps, SIGNAL(dotSizeChanged(double)), this,
              SLOT(handleDotSizeChanged(double)));
      connect(lineProps, SIGNAL(lineWidthChanged(double)), this,
              SLOT(handleLineWidthChanged(double)));
      connect(lineProps, SIGNAL(lineColorChanged(const QString &)), this,
              SLOT(handleLineColorChanged(const QString &)));
      connect(lineProps, SIGNAL(showLineSegmentChanged(bool)), this,
              SLOT(handleShowLineSegmentChanged(bool)));
    }
  }
}

ModelChartInfo *lcChartWizard::FindLineChartGraphIndex(
    const ViewLineChartProps *lps,
    const vector<pair<QWidget *, ModelChartInfo *>> &infos) {
  // Find the graph index for the given ViewLineChartProps object
  for (const auto &pair : infos) {
    if (pair.first == lps) {
      ModelChartInfo *info = pair.second;
      if (info) {
        return info;
        ;
      }
    }
  }
  return NULL; // Return -1 if not found
}

void lcChartWizard::handleShowLineSegmentChanged(bool checked) {
  ViewLineChartProps *lps = qobject_cast<ViewLineChartProps *>(sender());
  QCustomPlot *qcp = vcw->getQCustomPlot();
  if (lps) {
    ModelChartInfo *info = FindLineChartGraphIndex(lps, m_vWidModelChartInfo);
    if (!info) {
      return; // Exit if ModelChartInfo is not found
    }
    int graphIndex = info->GetGraphIndex();
    if (graphIndex < 0) {
      return; // Exit if graph index is not found
    }

    // Show or hide the line segment based on the checkbox state
    qcp->graph(graphIndex)->setLineStyle(
        checked ? QCPGraph::lsLine : QCPGraph::lsNone);
    qcp->replot(); // Replot to reflect changes
  }
}

void lcChartWizard::handleLineColorChanged(const QString &color) {
  ViewLineChartProps *lps = qobject_cast<ViewLineChartProps *>(sender());
  QCustomPlot *qcp = vcw->getQCustomPlot();
  if (lps) {
    ModelChartInfo *info = FindLineChartGraphIndex(lps, m_vWidModelChartInfo);
    if (!info) {
      return; // Exit if ModelChartInfo is not found
    }
    int graphIndex = info->GetGraphIndex();
    if (graphIndex < 0) {
      return; // Exit if graph index is not found
    }

    QPen pen = qcp->graph(graphIndex)->pen();
    pen.setColor(QColor(color)); // Set the new line color
    qcp->graph(graphIndex)->setPen(pen);
    qcp->replot(); // Replot to reflect changes
  }
}

void lcChartWizard::handleDotSizeChanged(double size) {
  ViewLineChartProps *lps = qobject_cast<ViewLineChartProps *>(sender());
  QCustomPlot *qcp = vcw->getQCustomPlot();
  if (lps) {

    ModelChartInfo *info = FindLineChartGraphIndex(lps, m_vWidModelChartInfo);
    if (!info) {
      return; // Exit if ModelChartInfo is not found
    }
    int graphIndex = info->GetGraphIndex();
    if (graphIndex < 0) {
      return; // Exit if graph index is not found
    }

    QCPScatterStyle currentStyle = qcp->graph(graphIndex)->scatterStyle();
    qcp->graph(graphIndex)
        ->setScatterStyle(QCPScatterStyle(currentStyle.shape(), size));
    qcp->replot(); // Replot to reflect changes
  }
}

void lcChartWizard::handleLineWidthChanged(double width) {
  ViewLineChartProps *lps = qobject_cast<ViewLineChartProps *>(sender());
  QCustomPlot *qcp = vcw->getQCustomPlot();
  if (lps) {
    ModelChartInfo *info = FindLineChartGraphIndex(lps, m_vWidModelChartInfo);
    if (!info) {
      return; // Exit if ModelChartInfo is not found
    }
    int graphIndex = info->GetGraphIndex();
    QPen pen = qcp->graph(graphIndex)->pen();
    pen.setWidthF(width); // Set the new line width
    qcp->graph(graphIndex)->setPen(pen);
    qcp->replot(); // Replot to reflect changes
  }
}

void lcChartWizard::handleDotStyleChanged(int idx) {
  ViewLineChartProps *lps = qobject_cast<ViewLineChartProps *>(sender());
  QCustomPlot *qcp = vcw->getQCustomPlot();
  QComboBox *cbb = lps->getDotStyleComboBox();
  if (!cbb) {
    return;
  }
  // get data that use addItem to add
  QCPScatterStyle::ScatterShape shape =
      static_cast<QCPScatterStyle::ScatterShape>(cbb->itemData(idx).toInt());
  // QCPScatterStyle::ScatterShape shape =
  // static_cast<QCPScatterStyle::ScatterShape>(idx);

  if (lps) {
    // Find the corresponding ModelChartInfo based on the sender
    ModelChartInfo *info = FindLineChartGraphIndex(lps, m_vWidModelChartInfo);
    if (!info) {
      return; // Exit if ModelChartInfo is not found
    }
    int graphIndex = info->GetGraphIndex();
    // get current scatter style size
    QCPScatterStyle currentStyle = qcp->graph(graphIndex)->scatterStyle();
    // set current style with new shape and current size
    qcp->graph(graphIndex)
        ->setScatterStyle(QCPScatterStyle(shape, currentStyle.size()));
    qcp->replot(); // Replot to reflect changes
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
  QCustomPlot *qcp = vcw->getQCustomPlot();
  if (lps) {
    lps->setTitle(tgtName); // Update the title of the ViewLineChartProps
    ModelChartInfo *info = FindLineChartGraphIndex(lps, m_vWidModelChartInfo);
    if (!info) {
      return; // Exit if ModelChartInfo is not found
    }
    info->SetLegendName(tgtName);
    // Update the legend name in the QCustomPlot
    int graphIndex = info->GetGraphIndex();
    qcp->graph(graphIndex)->setName(tgtName);
    qcp->replot(); // Replot to reflect changes
  }
}

void lcChartWizard::handleLineChartSelection() {
  QCustomPlot *qcp = (QCustomPlot *)sender();
  // Decide which line chart is selected
  if (qcp->selectedGraphs().isEmpty()) {
    return; // No graph selected
  }
  QCPGraph *selectedGraph = qcp->selectedGraphs().first();

  // find Props based on selectedGraph and m_vWidModelChartInfo
  QWidget *tgtWidget = NULL;
  for (const auto &pair : m_vWidModelChartInfo) {
    ModelChartInfo *ModelChartInfo = pair.second;
    int graphIdx = ModelChartInfo->GetGraphIndex();
    if (qcp->graph(graphIdx) == selectedGraph) {
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
      for (const auto &pair : m_vWidModelChartInfo) {
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

QWidget *lcChartWizard::CreateLineChartProps(ModelChartInfo *ModelChartInfo) {
  ViewLineChartProps *section = new ViewLineChartProps(
      ModelChartInfo ? ModelChartInfo->GetLegendName() : "Default Title", 33);

  QCustomPlot *qcp = vcw->getQCustomPlot();
  // Use ModelChartInfo to draw line chart
  if (ModelChartInfo) {

    ModelChartInfo->SetQCustomPlot(qcp);
    // use graphcount to calculate graph index and set

    qcp->addGraph();
    int graphCount = qcp->graphCount();
    int graphIndx = graphCount - 1; // Use the last graph index
    ModelChartInfo->SetGraphIndex(graphIndx);
    qcp->graph(graphIndx)->setData(*ModelChartInfo->GetX(),
                                   *ModelChartInfo->GetY());
    // set pen
    qcp->graph(graphIndx)->setPen(ModelChartInfo->GetPen());
    // set brush
    qcp->graph(graphIndx)->setBrush(ModelChartInfo->GetBrush());
    qcp->xAxis->setRange(0, ModelChartInfo->GetX()->back());
    qcp->yAxis->setRange(*std::min_element(ModelChartInfo->GetY()->begin(),
                                           ModelChartInfo->GetY()->end()),
                         *std::max_element(ModelChartInfo->GetY()->begin(),
                                           ModelChartInfo->GetY()->end()));
    // add scatter style to fill circle
    qcp->graph(graphIndx)->setScatterStyle(
        QCPScatterStyle(QCPScatterStyle::ssDisc, 5));
    // set line style
    // qcp->graph(graphIndx)->setLineStyle(QCPGraph::lsNone);
    qcp->graph(graphIndx)->setName(ModelChartInfo->GetLegendName());
    // set no line

    QPen graphPen = ModelChartInfo->GetPen();
    QColor graphColor = graphPen.color();
    QColor complementaryColor(255 - graphColor.red(), 255 - graphColor.green(),
                              255 - graphColor.blue());
    QCPSelectionDecorator *decorator = new QCPSelectionDecorator();
    decorator->setPen(
        QPen(complementaryColor, 2)); // Set selection color to green
    qcp->graph(graphIndx)->setSelectionDecorator(decorator);
  }
  return section;
}
