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


std::vector<QColor> jetColor(int numOfColorStepInJetColor) {
  std::vector<QColor> colors;
  colors.reserve(numOfColorStepInJetColor);
  for (int i = 0; i < numOfColorStepInJetColor; ++i) {
    double ratio = static_cast<double>(i) / (numOfColorStepInJetColor - 1);
    int r = static_cast<int>(255 * std::max(0.0, std::min(1.0, ratio * 4 - 1)));
    int g = static_cast<int>(255 * std::max(0.0, std::min(1.0, 2 - std::abs(ratio * 4 - 2))));
    int b = static_cast<int>(255 * std::max(0.0, std::min(1.0, 3 - ratio * 4)));
    colors.emplace_back(r, g, b);
  }
  colors.shrink_to_fit(); // Optional: shrink to fit for memory efficiency
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

  ModelTimeSequenceParser *tsp = new ModelTimeSequenceParser(this);
  tsp->OpenFile("/home/justin126/workspace/ShareTest/cw/ts.txt");
  tsp->start();
  tsp->Wait(); // Wait for the thread to finish

  vector<double> *timeStamps = tsp->GetTimeStamps();

  vector<pair<string, vector<pair<TimeSequencePair *, TimeSequencePair *>>>>
      *pairsByActId = tsp->GetTimeSequencePairsByActId();

  QCustomPlot *qcp = vcw->getQCustomPlot();
  QVBoxLayout *vlytLeftProps = vcw->getVLayoutLeftProps();

  double baseTime = (*timeStamps)[0];
  double apiBarHeight = 1;
  double apiSpacing = 1;
  vector<QColor> jetColors = jetColor(pairsByActId->size());
  QSharedPointer<QCPAxisTickerText> textTicker(new QCPAxisTickerText);
  QColor randomColor;
  double y1, yc, y2;
  for (size_t j = 0; j < pairsByActId->size(); j++) {
    const auto &pairs = (*pairsByActId)[j].second;

    // cout << (*pairsByActId)[j].first << ": ";

    if (pairs.size() > 0 && 1 <= pairs.size()) {
      const auto &pair = (pairs)[0];
      TimeSequencePair *recvPair = pair.first;
      string recvActIdStr = tsp->ActId2Str(recvPair->GetActId());
      randomColor =
          QColor(tsp->ActId2JetHexColor(recvPair->GetActId()).c_str());
      y1 = j * (apiBarHeight + apiSpacing); // y position based on index
      yc = y1 + apiBarHeight / 2;           // center y position for text
      y2 = y1 + apiBarHeight;               // height of the rectangle
      textTicker->addTick(yc, QString::fromStdString(recvActIdStr));
    }
    // cout << "ActId: " << pairs.first << " has "
    //      << pairs.second.size() << " pairs." << endl;
    for (size_t i = 0; i < pairs.size(); i++) {
      const auto &pair = (pairs)[i];
      TimeSequencePair *recvPair = pair.first;
      TimeSequencePair *sendPair = pair.second;

      // action id to string
      

      bool pairedApi = recvPair && sendPair;
      bool noPairedApi = recvPair && !sendPair;
      
      if (pairedApi) {
        // rect x from recvPair timestamp to sendPair timestamp
        // printf("%s ~ %s \n", recvPair->GetTimeStampStr().c_str(),
        //        sendPair->GetTimeStampStr().c_str());
        double x1 = recvPair->GetTimeStamp() - baseTime;
        double x2 = sendPair->GetTimeStamp() - baseTime;
        
        // printf("%f ~ %f, ", x1, x2);
        // Create a rectangle item for the API call
        // 
        // check x1 < x2, if not swap
        if (x1 > x2) {
          std::swap(x1, x2);
        }
        if (y1 > y2) {
          std::swap(y1, y2);
        }
        QCPItemRect *rect = new QCPItemRect(qcp);

        rect->setPen(
            QPen(randomColor, 1)); // Set random color for the rectangle
        rect->setBrush(
            QBrush(randomColor, Qt::SolidPattern)); // Fill with color
        rect->topLeft->setCoords(x1, y2);
        rect->bottomRight->setCoords(x2, y1);
        rect->setSelectable(QCP::stWhole); // Make the rectangle selectable
        rect->setSelectedPen(
            QPen(randomColor, 4,
                 Qt::DashLine)); // Thicker, red, dashed outline when selected
        rect->setSelectedBrush(QBrush(Qt::yellow)); // Yellow fill when selected

      } else if (noPairedApi) {
        // No paired API, draw a rectangle with a different color
        double x1 = recvPair->GetTimeStamp() - baseTime;
        // printf("%f , ", x1);
        // textTicker->addTick(yc, QString::fromStdString(recvActIdStr));
        // draw vertical line
        QCPItemLine *line = new QCPItemLine(qcp);
        line->setPen(QPen(randomColor, 1)); // Set random color for the line
        line->start->setCoords(x1, y1);
        line->end->setCoords(x1, y2);

        line->setSelectedPen(
            QPen(randomColor, 4,
                 Qt::DashLine)); // Thicker, red, dashed outline when selected
        // line->setSelectedBrush(QBrush(Qt::yellow));
      }
      // printf("\n");
    }
  }

  qcp->setInteractions(QCP::iSelectItems | QCP::iRangeDrag | QCP::iRangeZoom);
  connect(qcp, SIGNAL(itemClick(QCPAbstractItem *, QMouseEvent *)), this,
          SLOT(handleTimeSeqItemClick(QCPAbstractItem *, QMouseEvent *)));
  // connect mousePress with SIGNAL
  connect(qcp, SIGNAL(mousePress(QMouseEvent *)), this,
          SLOT(handleTimeSeqMousePressed(QMouseEvent *)));

  // // fit viewport to see all data
  qcp->xAxis->setRange(0, (*timeStamps)[timeStamps->size() - 1] - baseTime);
  qcp->yAxis->setRange(0, pairsByActId->size() * (apiBarHeight + apiSpacing));
  qcp->xAxis->setLabel("Time (ms)");
  qcp->yAxis->setTicker(textTicker);
  // qcp->xAxis->setTickLabelFont(QFont(QFont().family(), 8));
  // qcp->yAxis->setTickLabelFont(QFont(QFont().family(), 8));
  qcp->yAxis->setLabel("API");
  qcp->replot();
  // x axis milisecond
}

void lcChartWizard::handleTimeSeqMousePressed(QMouseEvent *event) {
  // Handle mouse press event
  QCustomPlot *qcp = (QCustomPlot *)sender();
  if (m_ToolTip) {
    m_ToolTip->setVisible(false); // Hide tooltip on mouse press
  }
  qcp->replot();
}

void lcChartWizard::handleTimeSeqItemClick(QCPAbstractItem *item,
                                           QMouseEvent *event) {
  // Handle item click event
  QCustomPlot *qcp = (QCustomPlot *)sender();
  double x = qcp->xAxis->pixelToCoord(event->pos().x());
  double y = qcp->yAxis->pixelToCoord(event->pos().y());

  if (!m_ToolTip) {
    m_ToolTip = new QCPItemText((QCustomPlot *)sender());
    m_ToolTip->setColor(Qt::black);
  }
  m_ToolTip->setVisible(false);
  m_ToolTip->position->setCoords(x, y - 1);

  if (QCPItemRect *rect = qobject_cast<QCPItemRect *>(item)) {

    double x1 = rect->topLeft->key();
    double y1 = rect->topLeft->value();
    double x2 = rect->bottomRight->key();
    double y2 = rect->bottomRight->value();

    QString info = QString("%1 ~ %2 ms").arg(x1).arg(x2);

    m_ToolTip->setText(info);

  } else if (QCPItemLine *line = qobject_cast<QCPItemLine *>(item)) {

    double x1 = line->start->key();
    double y1 = line->start->value();
    double x2 = line->end->key();
    double y2 = line->end->value();

    QString info = QString("%1 ms").arg(x1);
    m_ToolTip->setText(info);
  }
  if (!item) {
    m_ToolTip->setVisible(false);
  } else {
    // check item selected, if not m_ToolTip hide else show
    if (item->selected()) {
      m_ToolTip->setVisible(true);
    } else {
      m_ToolTip->setVisible(false);
    }
  }

  qcp->replot();
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
