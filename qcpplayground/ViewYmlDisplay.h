#ifndef VIEW_YML_DISPLAY_H
#define VIEW_YML_DISPLAY_H

#include <QtCore>
#include <QtWidgets>

#include <QWidget>
#include <iostream>
#include "qcustomplot.h"
#include <QVector>

using namespace std;

class ViewYmlDisplay : public QWidget
{
    Q_OBJECT

public:
    explicit ViewYmlDisplay(QWidget *parent = nullptr);

  void setupQuadraticDemo(QCustomPlot *customPlot);
  void setupSimpleDemo(QCustomPlot *customPlot);
  void setupSincScatterDemo(QCustomPlot *customPlot);
  void setupScatterStyleDemo(QCustomPlot *customPlot);
  void setupLineStyleDemo(QCustomPlot *customPlot);
  void setupScatterPixmapDemo(QCustomPlot *customPlot);
  void setupDateDemo(QCustomPlot *customPlot);
  void setupTextureBrushDemo(QCustomPlot *customPlot);
  void setupMultiAxisDemo(QCustomPlot *customPlot);
  void setupLogarithmicDemo(QCustomPlot *customPlot);
  void setupRealtimeDataDemo(QCustomPlot *customPlot);
  void setupParametricCurveDemo(QCustomPlot *customPlot);
  void setupBarChartDemo(QCustomPlot *customPlot);
  void setupStatisticalDemo(QCustomPlot *customPlot);
  void setupSimpleItemDemo(QCustomPlot *customPlot);
  void setupItemDemo(QCustomPlot *customPlot);
  void setupStyledDemo(QCustomPlot *customPlot);
  void setupAdvancedAxesDemo(QCustomPlot *customPlot);
  void setupColorMapDemo(QCustomPlot *customPlot);
  void setupFinancialDemo(QCustomPlot *customPlot);
void setupPolarPlotDemo(QCustomPlot *customPlot);
    QTreeWidget* twYmlDisplay;
    QSplitter* spltMain;
    QTextEdit*  teManual;
    QWidget* widQcp;

private:
    void Widgets();
    void Layouts();
    QString demoName;
    QTimer dataTimer;
  QCPItemTracer *itemDemoPhaseTracer;
  QCustomPlot *qcp=NULL;
   QCPItemRect* rect1;
    QCPItemEllipse* wafer;
    QCPItemRect *rectItem;

    void interactWaferMap(QCustomPlot* qcp);
    void overlapWaferMapAndContour(QCustomPlot* qcp);

private slots:
    void onComboBoxChanged(int idx);
    void handleQcpPress(QMouseEvent*);
    void handleQcpItemClick(QCPAbstractItem *item, QMouseEvent *event);
   

};

#endif /* VIEW_YML_DISPLAY_H */