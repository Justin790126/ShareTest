#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "qcustomplot.h"

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private:
    QCustomPlot *customPlot;

    // Blackman-windowed sinc interpolation function
    std::vector<double> blackmanSincInterpolation(const std::vector<double>& x,
                                                 const std::vector<double>& value,
                                                 const std::vector<double>& xq,
                                                 int windowSize = 8);
};

#endif // MAINWINDOW_H