#ifndef LC_CHARTWIZARD_H
#define LC_CHARTWIZARD_H

#include "ViewLineChartProps.h"
#include "qcustomplot.h"
#include <QtGui>
#include <iostream>

#include "data.h"

using namespace std;

const extern std::vector<double> SLICE_DATA;

/*
 * ChartInfo
 * This class is responsible for storing information about a chart.
 * It includes methods to set the x and y data, pen, brush, and graph index.
 */
class ChartInfo {
public:
  ChartInfo() = default;
  ~ChartInfo() = default;
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

  friend ofstream &operator<<(std::ofstream &ofs, const ChartInfo &info) {

    return ofs;
  }

private:
  QVector<double> m_qvX;
  QVector<double> m_qvY;

  QString m_sLegendName;

  QPen m_pen;
  QBrush m_brush;
  int m_iGphIdx;
  QCustomPlot *m_pQCP = NULL;
};

/*
 * lcChartWizard
 * This class is responsible for creating a chart wizard interface.
 * It includes methods to create line chart properties and manage the layout.
 */

class lcChartWizard : public QWidget {
  Q_OBJECT
public:
  lcChartWizard(QWidget *parent = nullptr);
  ~lcChartWizard();
  QWidget *CreateLineChartProps(ChartInfo *chartInfo);
  QVBoxLayout *vlytLeft;
  QVBoxLayout *vlytLeftProps;

private:
  void Widgets();
  void Layouts();
  void UI();
  void ConnectLineChartProps();

  vector<pair<QWidget *, ChartInfo *>> m_vWidChartInfo;

  QCustomPlot *m_qcp;

private slots:
    void handleLineChartSelection();

    void handleLineNameChanged(const QString &name);
};

#endif /* LC_CHARTWIZARD_H */