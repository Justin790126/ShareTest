#include "lcChartWizard.h"
#include "algorithm"



inline std::vector<double>
find_all_x_by_linear_interp(const std::vector<double> &x_nm,
                            const std::vector<double> &y, double threshold) {
  if (x_nm.size() != y.size() || x_nm.empty())
    return {}; // Return empty vector if sizes do not match or are empty

  std::vector<double> result;
  result.reserve(x_nm.size()); // Reserve space for efficiency
  result.emplace_back(x_nm[0]);
  for (size_t i = 1; i < y.size(); ++i) {
    bool crosses = ((y[i - 1] < threshold && y[i] >= threshold) ||
                    (y[i - 1] > threshold && y[i] <= threshold));
    if (crosses) {
      double x1 = x_nm[i - 1], x2 = x_nm[i];
      double y1 = y[i - 1], y2 = y[i];
      if (std::abs(y2 - y1) < 1e-12) // avoid division by zero
        result.push_back(0.5 * (x1 + x2));
      else
        result.push_back(x1 + (threshold - y1) * (x2 - x1) / (y2 - y1));
    }
  }
  result.push_back(x_nm.back()); // Add the last x value
  result.shrink_to_fit(); // Optional: shrink to fit for memory efficiency
  return result;
}

std::vector<QColor> jetColor(int n) {
    std::vector<QColor> colors;
    colors.reserve(n);
    for (int i = 0; i < n; ++i) {
        double v = n == 1 ? 0.0 : double(i) / double(n - 1);

        double r = 1.5 - std::abs(4.0 * v - 3.0);
        if (r < 0.0) r = 0.0;
        if (r > 1.0) r = 1.0;

        double g = 1.5 - std::abs(4.0 * v - 2.0);
        if (g < 0.0) g = 0.0;
        if (g > 1.0) g = 1.0;

        double b = 1.5 - std::abs(4.0 * v - 1.0);
        if (b < 0.0) b = 0.0;
        if (b > 1.0) b = 1.0;

        colors.push_back(QColor(int(r * 255), int(g * 255), int(b * 255)));
    }
    return colors;
}

lcChartWizard::lcChartWizard(QWidget *parent) {
  // UI();
  vcw = new ViewChartWizard(parent);

  vcw->show();

#if TEST_SLICE_FEATURES
  QCustomPlot *qcp = vcw->getQCustomPlot();
  QVBoxLayout *vlytLeftProps = vcw->getVLayoutLeftProps();

  vector<double> x_nms(SLICE_DATA.size());
  double nmPerStep = 4;
  for (int i = 0; i < SLICE_DATA.size(); ++i) {
    x_nms[i] = i * nmPerStep; // Assuming x-axis is just the index
  }

  vector<double> cross_x = find_all_x_by_linear_interp(
      x_nms, SLICE_DATA, 0.5); // Find all x where y crosses 0.001

  vector<double> y_noise(SLICE_DATA.size());
  for (int i = 0; i < SLICE_DATA.size(); ++i) {
    // add random noise
    y_noise[i] =
        SLICE_DATA[i] + ((rand() % 100) / 10000.0); // Adding small random noise
  }
  vector<double> cross_x_noise = find_all_x_by_linear_interp(
      x_nms, y_noise, 0.5); // Find all x where noisy y crosses 0.001

  ModelChartInfo *info = new ModelChartInfo();
  info->SetX(QVector<double>::fromStdVector(x_nms));
  info->SetY(QVector<double>::fromStdVector(y_noise));
  info->SetIntersectionX(QVector<double>::fromStdVector(cross_x_noise));
  info->SetThreshold(0.5);         // Set the threshold for intersection
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
  info2->SetIntersectionX(QVector<double>::fromStdVector(cross_x));
  info2->SetThreshold(0.5);        // Set the threshold for intersection
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
#endif

  // create jet color lookup table in vector<QColor>

  


  ModelTimeSequenceParser* tsp = new ModelTimeSequenceParser(this);
  tsp->OpenFile("/home/justin126/workspace/ShareTest/cw/ts.txt");
  tsp->start();
  tsp->Wait(); // Wait for the thread to finish

  vector<pair<TimeSequencePair*, TimeSequencePair*>> *pairs =
      tsp->GetTimeSequencePairs();
  vector<double> *timeStamps = tsp->GetTimeStamps();


  QCustomPlot *qcp = vcw->getQCustomPlot();
  QVBoxLayout *vlytLeftProps = vcw->getVLayoutLeftProps();

  double baseTime = (*timeStamps)[0];
  double apiBarHeight = 1;
  double apiSpacing = 1;
  vector<QColor> jetColors = jetColor(pairs->size());
  for (size_t i = 0; i < pairs->size(); i++) {
    const auto &pair = (*pairs)[i];
    TimeSequencePair *recvPair = pair.first;
    TimeSequencePair *sendPair = pair.second;

    bool pairedApi = recvPair && sendPair;
    bool noPairedApi = recvPair && !sendPair;
    QColor randomColor = QColor(tsp->ActId2JetHexColor(recvPair->GetActId()).c_str());
    if (pairedApi) {
      // rect x from recvPair timestamp to sendPair timestamp
      double x1 = recvPair->GetTimeStamp() - baseTime;
      double x2 = sendPair->GetTimeStamp() - baseTime;
      double y1 = i * (apiBarHeight + apiSpacing); // y position based on index
      double y2 = y1 + apiBarHeight; // height of the rectangle
      // Create a rectangle item for the API call
      
      // check x1 < x2, if not swap
      if (x1 > x2) {
        std::swap(x1, x2);
      }
      if (y1 > y2) {
        std::swap(y1, y2);
      }
      QCPItemRect *rect = new QCPItemRect(qcp);
      
      
      rect->setPen(QPen(randomColor, 1)); // Set random color for the rectangle
      rect->setBrush(QBrush(randomColor, Qt::SolidPattern)); // Fill with color
      rect->topLeft->setCoords(x1, y2);
      rect->bottomRight->setCoords(x2, y1);
      rect->setSelectable(QCP::stWhole); // Make the rectangle selectable
      
      
    } else if (noPairedApi) {
      // No paired API, draw a rectangle with a different color
      double x1 = recvPair->GetTimeStamp() - baseTime;
      double y1 = i * (apiBarHeight + apiSpacing);
      double y2 = y1 + apiBarHeight; // height of the rectangle
      double centerY = (y1 + y2) / 2.0; // center of the rectangle
      double crossLen = 0.05;

      double crossLeftBottomX = x1 - crossLen * std::cos(M_PI / 3);
      double crossLeftBottomY = centerY - crossLen * std::sin(M_PI / 3);
      double crossRightTopX = x1 + crossLen * std::cos(M_PI / 3);
      double crossRightTopY = centerY + crossLen * std::sin(M_PI / 3);

      // draw vertical line
      QCPItemLine *line = new QCPItemLine(qcp);
      line->setPen(QPen(randomColor, 1)); // Set random color for the line
      line->start->setCoords(x1, y1);
      line->end->setCoords(x1, y2);

      QCPItemLine *crossLeft = new QCPItemLine(qcp);
      crossLeft->setPen(QPen(randomColor, 1)); // Set random color for the line
      crossLeft->start->setCoords(crossLeftBottomX, crossLeftBottomY);
      crossLeft->end->setCoords(crossRightTopX, crossRightTopY);

      QCPItemLine *crossRight = new QCPItemLine(qcp);
      crossRight->setPen(QPen(randomColor, 1)); // Set random color for the line
      crossRight->start->setCoords(crossRightTopX, crossLeftBottomY);
      crossRight->end->setCoords(crossLeftBottomX, crossRightTopY);

    }
  }

  
  // fit viewport to see all data
  qcp->xAxis->setRange(0, (*timeStamps)[timeStamps->size() - 1] - baseTime);
  qcp->yAxis->setRange(0, pairs->size() * (apiBarHeight + apiSpacing));
  qcp->replot();
  // x axis milisecond
  qcp->xAxis->setLabel("Time (ms)");

}

void lcChartWizard::ConnectGeneralProps() {
  ViewChartProps *vcpGeneral = vcw->getVCPGeneral();
  if (vcpGeneral) {
    connect(vcpGeneral, SIGNAL(chartTitleChanged(const QString &)), this,
            SLOT(handleGeneralTitleChanged(const QString &)));
    connect(vcpGeneral, SIGNAL(xLabelChanged(const QString &)), this,
            SLOT(handleGeneralXLabelChanged(const QString &)));
    connect(vcpGeneral, SIGNAL(yLabelChanged(const QString &)), this,
            SLOT(handleGeneralYLabelChanged(const QString &)));
    connect(vcpGeneral, SIGNAL(legendVisibilityChanged(bool)), this,
            SLOT(handleGeneralLegendVisibilityChanged(bool)));
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

void lcChartWizard::handleGeneralXLabelChanged(const QString &label) {
  QCustomPlot *qcp = vcw->getQCustomPlot();
  if (qcp) {
    qcp->xAxis->setLabel(label);
    qcp->replot(); // Replot to reflect changes
  }
}

void lcChartWizard::handleGeneralYLabelChanged(const QString &label) {
  QCustomPlot *qcp = vcw->getQCustomPlot();
  if (qcp) {
    qcp->yAxis->setLabel(label);
    qcp->replot(); // Replot to reflect changes
  }
}

void lcChartWizard::handleGeneralLegendVisibilityChanged(bool visible) {
  QCustomPlot *qcp = vcw->getQCustomPlot();
  if (qcp) {
    qcp->legend->setVisible(visible);
    qcp->replot(); // Replot to reflect changes
  }
}

void lcChartWizard::ConnectLineChartProps() {
  // Connect signals from ViewLineChartProps to lcChartWizard
  for (const auto &pair : m_vWidModelChartInfo) {
    ViewLineChartProps *lineProps =
        qobject_cast<ViewLineChartProps *>(pair.first);
    if (lineProps) {
      connect(lineProps, SIGNAL(showGraphChanged(bool)), this,
              SLOT(handleShowGraphChanged(bool)));
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
      connect(lineProps, SIGNAL(showThresholdAndMetrologyChanged(bool)), this,
              SLOT(handleShowThresholdAndMetrologyChanged(bool)));
      connect(lineProps, SIGNAL(thresholdValueChanged(double)), this,
              SLOT(handleThresholdValueChanged(double)));
      connect(lineProps, SIGNAL(thresholdColorButtonClicked()), this,
              SLOT(handleThresholdColorButtonClicked()));
      connect(lineProps, SIGNAL(lineColorButtonClicked()), this,
              SLOT(handleLineColorButtonClicked()));
    }
  }
}

void lcChartWizard::handleThresholdColorButtonClicked() {
  // Handle the threshold color button click event
  ViewLineChartProps *lps = qobject_cast<ViewLineChartProps *>(sender());
  cout << "Threshold color button clicked" << endl;

  // init random QPen, QBrush and set to line
  QCustomPlot *qcp = vcw->getQCustomPlot();
  if (lps) {
    ModelChartInfo *info = FindLineChartGraphIndex(lps, m_vWidModelChartInfo);
    if (!info) {
      return; // Exit if ModelChartInfo is not found
    }
    int thresGraphIndex = info->GetThresholdGraphIndex();
    if (thresGraphIndex < 0) {
      return; // Exit if threshold graph index is not found
    }

    // Set a random color for the threshold line
    QPen pen = qcp->graph(thresGraphIndex)->pen();
    QColor newColor(rand() % 256, rand() % 256, rand() % 256);
    pen.setColor(newColor);
    qcp->graph(thresGraphIndex)->setPen(pen);
    // set random brush with random pattern
    QBrush brush = qcp->graph(thresGraphIndex)->brush();
    brush.setColor(newColor);
    brush.setStyle(static_cast<Qt::BrushStyle>(rand() % 6 + 1)); // Random style
    qcp->graph(thresGraphIndex)->setBrush(brush);
    qcp->replot(); // Replot to reflect changes
  }
}

void lcChartWizard::handleLineColorButtonClicked() {
  // Handle the line color button click event
  ViewLineChartProps *lps = qobject_cast<ViewLineChartProps *>(sender());
  cout << "Line color button clicked" << endl;
  // init random QPen, QBrush and set to line
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

    // Set a random color for the line
    QPen pen = qcp->graph(graphIndex)->pen();
    QColor newColor(rand() % 256, rand() % 256, rand() % 256);
    pen.setColor(newColor);
    qcp->graph(graphIndex)->setPen(pen);
    // set random brush with random pattern
    QBrush brush = qcp->graph(graphIndex)->brush();
    brush.setColor(newColor);
    brush.setStyle(static_cast<Qt::BrushStyle>(rand() % 6 + 1)); // Random style
    qcp->graph(graphIndex)->setBrush(brush);
    qcp->replot(); // Replot to reflect changes
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
      }
    }
  }
  return NULL; // Return -1 if not found
}

void lcChartWizard::handleThresholdValueChanged(double value) {
  ViewLineChartProps *lps = qobject_cast<ViewLineChartProps *>(sender());
  QCustomPlot *qcp = vcw->getQCustomPlot();
  if (lps) {
    ModelChartInfo *info = FindLineChartGraphIndex(lps, m_vWidModelChartInfo);
    if (!info) {
      return;
    }
    int thresGraphIndex = info->GetThresholdGraphIndex();
    if (thresGraphIndex < 0) {
      return;
    }

    // Update the threshold line with the new value
    qcp->graph(thresGraphIndex)->data()->clear();
    qcp->graph(thresGraphIndex)->addData(0, value);
    qcp->graph(thresGraphIndex)->addData(info->GetX()->back(), value);

    info->SetThreshold(value); // Update the threshold in ModelChartInfo

    // get current x,y
    QVector<double> *x = info->GetX();
    QVector<double> *y = info->GetY();
    // recalculate intersection points
    QVector<double> *intersectionX = info->GetIntersectionX();
    intersectionX->clear();
    // Find all x where y crosses the threshold
    QVector<double> newIntersectionX = QVector<double>::fromStdVector(
        find_all_x_by_linear_interp(x->toStdVector(), y->toStdVector(), value));
    *intersectionX = newIntersectionX; // Update intersection points

    // Recalculate metrology text items based on new intersection points
    AddMetrologyTextItem(info, qcp);

    qcp->replot(); // Replot to reflect changes
  }
}

void lcChartWizard::handleShowThresholdAndMetrologyChanged(bool checked) {
  ViewLineChartProps *lps = qobject_cast<ViewLineChartProps *>(sender());
  QCustomPlot *qcp = vcw->getQCustomPlot();
  if (lps) {
    ModelChartInfo *info = FindLineChartGraphIndex(lps, m_vWidModelChartInfo);
    if (!info) {
      return; // Exit if ModelChartInfo is not found
    }
    int thresGraphIndex = info->GetThresholdGraphIndex();
    if (thresGraphIndex > 0) {
      qcp->graph(thresGraphIndex)->setVisible(checked);
    }

    // Show or hide the threshold line and metrology text items based on the
    // checkbox state
    for (int i = 0; i < info->GetMetrologyTextItems()->size(); ++i) {
      QCPItemText *textItem = (*info->GetMetrologyTextItems())[i];
      textItem->setVisible(checked);
    }

    qcp->replot(); // Replot to reflect changes
  }
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
    qcp->graph(graphIndex)
        ->setLineStyle(checked ? QCPGraph::lsLine : QCPGraph::lsNone);
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

void lcChartWizard::handleShowGraphChanged(bool checked) {
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

    // Show or hide the graph based on the checkbox state
    qcp->graph(graphIndex)->setVisible(checked);
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

void lcChartWizard::AddMetrologyTextItem(ModelChartInfo *info,
                                         QCustomPlot *qcp) {

  info->ClearMetrologyTextItems();
  info->ReserveMetrologyTextItems(info->GetIntersectionX()->size() - 1);
  QVector<double> *intersectionX = info->GetIntersectionX();
  for (int i = 0; i < (int)intersectionX->size() - 1; ++i) {
    double x1 = (*intersectionX)[i];
    double x2 = (*intersectionX)[i + 1];
    double cd = fabs(x2 - x1);
    double txtPos = x1 + cd / 2.0; // Position text in the middle of the segment
    if (cd < 1e-6) {
      continue; // Skip if the distance is too small
    }
    // draw text items with content of cd

    ViewScalableItemText *textItem = new ViewScalableItemText(qcp);
    textItem->setPositionAlignment(Qt::AlignCenter);
    textItem->position->setCoords(txtPos, info->GetThreshold() + 0.01);
    textItem->setText(QString().sprintf("CD=%.6fnm", cd));
    textItem->setFont(QFont("Arial", 10));
    textItem->setColor(Qt::black); // Set text color to black
    textItem->setLayer("overlay"); // Set layer to overlay
    textItem->setScalingAxis(qcp->xAxis, qcp->xAxis->range().size());
    textItem->setVisible(false);

    info->AddMetrologyTextItem(textItem);
  }
  info->ShrinkMetrologyTextItems();
}

QWidget *lcChartWizard::CreateLineChartProps(ModelChartInfo *info) {
  ViewLineChartProps *section = new ViewLineChartProps(
      info ? info->GetLegendName() : "Default Title", 33);

  QCustomPlot *qcp = vcw->getQCustomPlot();
  // Use info to draw line chart
  if (info) {

    info->SetQCustomPlot(qcp);

    // draw signals
    qcp->addGraph();
    int graphCount = qcp->graphCount();
    int graphIndx = graphCount - 1;
    info->SetGraphIndex(graphIndx);
    qcp->graph(graphIndx)->setData(*info->GetX(), *info->GetY());
    qcp->graph(graphIndx)->setPen(info->GetPen());
    qcp->graph(graphIndx)->setBrush(info->GetBrush());
    qcp->xAxis->setRange(0, info->GetX()->back());
    qcp->yAxis->setRange(
        *std::min_element(info->GetY()->begin(), info->GetY()->end()),
        *std::max_element(info->GetY()->begin(), info->GetY()->end()));
    qcp->graph(graphIndx)->setScatterStyle(
        QCPScatterStyle(QCPScatterStyle::ssDisc, 5));
    qcp->graph(graphIndx)->setName(info->GetLegendName());

    QPen graphPen = info->GetPen();
    QColor graphColor = graphPen.color();
    QColor complementaryColor(255 - graphColor.red(), 255 - graphColor.green(),
                              255 - graphColor.blue());

    // draw threshold line
    QCPGraph *gphIntersectThreshold = qcp->addGraph();
    graphCount = qcp->graphCount();
    int thresGraphIndx = graphCount - 1;
    info->SetThresholdGraphIndex(thresGraphIndx);
    gphIntersectThreshold->setPen(QPen(complementaryColor, 3));
    QString thresholdName;
    thresholdName.sprintf("%s Threshold",
                          info ? info->GetLegendName().toStdString().c_str()
                               : "Default");
    gphIntersectThreshold->setName(thresholdName);
    gphIntersectThreshold->addData(0, info->GetThreshold());
    gphIntersectThreshold->addData(info->GetX()->back(), info->GetThreshold());
    gphIntersectThreshold->setVisible(false);

    // draw CD metrology
    AddMetrologyTextItem(info, qcp);

    QCPSelectionDecorator *decorator = new QCPSelectionDecorator();
    decorator->setPen(
        QPen(complementaryColor, 2)); // Set selection color to green
    qcp->graph(graphIndx)->setSelectionDecorator(decorator);
  }
  return section;
}
