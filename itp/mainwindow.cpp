#include "mainwindow.h"
#include <QVector>
#include <cmath>

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent) {
    // Create QCustomPlot widget
    customPlot = new QCustomPlot(this);
    setCentralWidget(customPlot);
    setGeometry(100, 100, 800, 600);

    // Sample data (e.g., representing a lithography intensity profile)
    std::vector<double> x = {0.0, 1.0, 2.0, 3.0, 4.0};
    std::vector<double> value = {1.0, 2.0, 0.0, 2.0, 1.0};

    // Query points for interpolation (denser grid for smooth curve)
    std::vector<double> xq;
    for (double i = 0.0; i <= 4.0; i += 0.1) {
        xq.push_back(i);
    }

    // Perform Blackman-windowed sinc interpolation
    std::vector<double> iptvals = blackmanSincInterpolation(x, value, xq);

    // Convert std::vector to QVector for QCustomPlot
    QVector<double> qx, qvalue, qxq, qiptvals;
    for (double val : x) qx.push_back(val);
    for (double val : value) qvalue.push_back(val);
    for (double val : xq) qxq.push_back(val);
    for (double val : iptvals) qiptvals.push_back(val);

    // Plot original data points
    customPlot->addGraph();
    customPlot->graph(0)->setData(qx, qvalue);
    customPlot->graph(0)->setPen(QPen(Qt::blue));
    customPlot->graph(0)->setLineStyle(QCPGraph::lsNone);
    customPlot->graph(0)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssCircle, 8));
    customPlot->graph(0)->setName("Original Data");

    // Plot interpolated signal
    customPlot->addGraph();
    customPlot->graph(1)->setData(qxq, qiptvals);
    customPlot->graph(1)->setPen(QPen(Qt::red));
    customPlot->graph(1)->setName("Interpolated Signal");

    // Configure plot
    customPlot->xAxis->setLabel("Position (μm)");
    customPlot->yAxis->setLabel("Intensity");
    customPlot->xAxis->setRange(-0.5, 4.5);
    customPlot->yAxis->setRange(-0.5, 2.5);
    customPlot->legend->setVisible(true);
    customPlot->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom | QCP::iSelectPlottables);

    // Rescale axes to fit data
    customPlot->rescaleAxes();
    customPlot->replot();
}

MainWindow::~MainWindow() {}

std::vector<double> MainWindow::blackmanSincInterpolation(const std::vector<double>& x,
                                                         const std::vector<double>& value,
                                                         const std::vector<double>& xq,
                                                         int windowSize) {
    if (x.size() != value.size() || x.empty()) {
        return {};
    }

    std::vector<double> iptvals(xq.size(), 0.0);
    const double a0 = 0.42, a1 = 0.50, a2 = 0.08;

    for (size_t i = 0; i < xq.size(); ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < x.size(); ++j) {
            double tau = (xq[i] - x[j]);
            if (std::abs(tau) < 1e-10) {
                sum = value[j];
                break;
            }

            double piTau = M_PI * tau;
            double sinc = std::sin(piTau) / piTau;

            double w = 0.0;
            int halfWindow = windowSize / 2;
            if (std::abs(tau) <= halfWindow) {
                double n = tau + halfWindow;
                w = a0 - a1 * std::cos(2.0 * M_PI * n / windowSize) +
                    a2 * std::cos(4.0 * M_PI * n / windowSize);
            }

            sum += value[j] * sinc * w;
        }
        iptvals[i] = sum;
    }

    return iptvals;
}