#ifndef LC_CHARTWIZARD_H
#define LC_CHARTWIZARD_H

#include "ViewChartWizard.h"
#include "ViewLineChartProps.h"
#include "ViewScalableItemText.h"
#include "ViewTimeSeqItems.h"
#include "ModelTimeSequenceParser.h"
#include "qcustomplot.h"
#include <QtGui>
#include <iostream>

#include "data.h"

using namespace std;

const extern std::vector<double> SLICE_DATA;

/*
 * ModelChartInfo
 * This class is responsible for storing information about a chart.
 * It includes methods to set the x and y data, pen, brush, and graph index.
 */
class ModelChartInfo {
public:
  ModelChartInfo() = default;
  ~ModelChartInfo() = default;
  void SetX(const QVector<double> &qvX) { m_qvX = qvX; }
  QVector<double> *GetX() { return &m_qvX; }
  void SetY(const QVector<double> &qvY) { m_qvY = qvY; }
  QVector<double> *GetY() { return &m_qvY; }
  void SetPen(const QPen &pen) { m_pen = pen; }
  QPen GetPen() const { return m_pen; }
  void SetBrush(const QBrush &brush) { m_brush = brush; }
  QBrush GetBrush() const { return m_brush; }
  void SetGraphIndex(int idx) { m_iGphIdx = idx; }
  int GetGraphIndex() const { return m_iGphIdx; }
  void SetQCustomPlot(QCustomPlot *pQCP) { m_pQCP = pQCP; }
  QCustomPlot *GetQCustomPlot() const { return m_pQCP; }
  void SetLegendName(const QString &sLegendName) {
    m_sLegendName = sLegendName;
  }
  QString GetLegendName() const { return m_sLegendName; }
  void SetIntersectionX(const QVector<double> &qvIntersectionX) {
    m_qvIntersectionX = qvIntersectionX;
  }
  QVector<double> *GetIntersectionX() { return &m_qvIntersectionX; }
  void SetThreshold(double dThreshold) { m_dThreshold = dThreshold; }
  double GetThreshold() const { return m_dThreshold; }
  void SetThresholdGraphIndex(int idx) { m_iThresGphIdx = idx; }
  int GetThresholdGraphIndex() const { return m_iThresGphIdx; }
  void ReserveMetrologyTextItems(int size) {
    m_vMetrologyTextItems.reserve(size);
  }
  void AddMetrologyTextItem(QCPItemText *item) {
    m_vMetrologyTextItems.push_back(item);
  }
  void ShrinkMetrologyTextItems() {
    m_vMetrologyTextItems.shrink_to_fit();
  }
  void ClearMetrologyTextItems() {
    for (auto item : m_vMetrologyTextItems) {
      // if (item == nullptr) continue;
      if (m_pQCP) {
        m_pQCP->removeItem(item);
      }
      // delete item;
    }
    m_vMetrologyTextItems.clear();
  }
  vector<QCPItemText *> *GetMetrologyTextItems() {
    return &m_vMetrologyTextItems;
  }

  friend ofstream &operator<<(std::ofstream &ofs, const ModelChartInfo &info) {

    return ofs;
  }

private:
  QVector<double> m_qvX;
  QVector<double> m_qvY;
  QVector<double> m_qvIntersectionX;
  double m_dThreshold = 0.5;

  QString m_sLegendName;

  QPen m_pen;
  QBrush m_brush;
  int m_iGphIdx = -1; // signal
  int m_iThresGphIdx = -1; // threshold line
  vector<QCPItemText*> m_vMetrologyTextItems;
  QCustomPlot *m_pQCP = NULL;
};

/*
 * lcChartWizard
 * This class is responsible for creating a chart wizard interface.
 * It includes methods to create line chart properties and manage the layout.
 */

class lcChartWizard : public QObject {
  Q_OBJECT
public:
  lcChartWizard(QWidget *parent = nullptr);
  ~lcChartWizard();
  QWidget *CreateLineChartProps(ModelChartInfo *ModelChartInfo);

private:
  void ConnectLineChartProps();
  void ConnectGeneralProps();

  vector<pair<QWidget *, ModelChartInfo *>> m_vWidModelChartInfo;

  ViewChartWizard *vcw = NULL;

  ModelChartInfo *FindLineChartGraphIndex(
      const ViewLineChartProps *lps,
      const vector<pair<QWidget *, ModelChartInfo *>> &infos);

  void AddMetrologyTextItem(
      ModelChartInfo *modelChartInfo, QCustomPlot *qcp);

  QCPItemText* m_ToolTip = NULL;
private slots:
  void handleGeneralTitleChanged(const QString &title);
  void handleGeneralXLabelChanged(const QString &label);
  void handleGeneralYLabelChanged(const QString &label);
  void handleGeneralLegendVisibilityChanged(bool visible);

  void handleLineChartSelection();
  void handleShowGraphChanged(bool checked);
  void handleLineNameChanged(const QString &name);
  void handleDotStyleChanged(int idx);
  void handleDotSizeChanged(double size);
  void handleShowLineSegmentChanged(bool checked);
  void handleLineWidthChanged(double width);
  void handleLineColorChanged(const QString &color);
  void handleLineColorButtonClicked();
  void handleShowThresholdAndMetrologyChanged(bool checked);
  void handleThresholdValueChanged(double value);
  void handleThresholdColorButtonClicked();

  void handleTimeSeqItemClick(QCPAbstractItem*, QMouseEvent*);
  void handleTimeSeqMousePressed(QMouseEvent *event);
};

#endif /* LC_CHARTWIZARD_H */