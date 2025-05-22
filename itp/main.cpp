#include <QApplication>
#include <QMainWindow>
#include <QVector>
#include "qcustomplot.h"
#include <cmath>
#include <vector>
#include <string>
#include <iostream>

using namespace std;

double sinc(double x) {
    if (std::abs(x) < 1e-10) return 1.0; // Avoid division by zero
    return std::sin(M_PI * x) / (M_PI * x);
}

double windowFunction(double t, double M, const std::string& type) {
    if (std::abs(t) > M) return 0.0;
    if (type == "rectangular") {
        return 1.0;
    } else if (type == "hamming") {
        return 0.54 + 0.46 * std::cos(M_PI * t / M);
    } else if (type == "blackman") {
        return 0.42 + 0.5 * std::cos(M_PI * t / M) + 0.08 * std::cos(2 * M_PI * t / M);
    }
    throw std::invalid_argument("Unknown window type: " + type);
}

QVector<double> sincInterp(const QVector<double>& x, const QVector<double>& t, const QVector<double>& t_new, int M, const std::string& windowType) {
    double delta_x = t[1] - t[0]; // Spatial sampling interval (nm)
    QVector<double> x_new(t_new.size());

    for (int i = 0; i < t_new.size(); ++i) {
        double xn = t_new[i]; // Position in nm
        int n_start = std::floor(xn / delta_x - M);
        int n_end = std::floor(xn / delta_x + M);
        double sum_val = 0.0;

        for (int n = std::max(0, n_start); n <= std::min(x.size() - 1, n_end); ++n) {
            double tau = xn - n * delta_x;
            double sinc_val = sinc(tau / delta_x);
            double w = windowFunction(tau / delta_x, M, windowType);
            double h = sinc_val * w;
            sum_val += x[n] * h;
        }
        x_new[i] = sum_val;
    }
    return x_new;
}

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    QMainWindow window;

    // Generate sample intensity data for lithography (Gaussian-like profile)
    const int N = 51; // 51 samples from 0 to 500 nm
    QVector<double> t(N); // Sample positions (nm)
    QVector<double> x(N); // Intensity values
    double delta_x = 10.0; // Spatial sampling interval (nm)
    for (int i = 0; i < N; ++i) {
        t[i] = i * delta_x; // Positions: 0, 10, 20, ..., 500 nm
        double pos = t[i] - 250.0; // Center at 250 nm
        x[i] = 0.9 * std::exp(-pos * pos / (2 * 50.0 * 50.0)); // Gaussian intensity, sigma = 50 nm
    }

    // Generate finer points for interpolation
    const int N_new = 1001; // 0 to 500 nm with step 0.5 nm
    QVector<double> t_new(N_new);
    for (int i = 0; i < N_new; ++i) {
        t_new[i] = i * 0.5;
    }

    // Perform interpolation for each window
    int M = 4; // Half-window size
    QVector<double> x_rect = sincInterp(x, t, t_new, M, "rectangular");
    QVector<double> x_hamm = sincInterp(x, t, t_new, M, "hamming");
    QVector<double> x_black = sincInterp(x, t, t_new, M, "blackman");

    // Set up QCustomPlot
    QCustomPlot *customPlot = new QCustomPlot(&window);
    window.setCentralWidget(customPlot);
    window.resize(1000, 600);

    // Plot original intensity samples
    customPlot->addGraph();
    customPlot->graph(0)->setData(t, x);
    customPlot->graph(0)->setPen(QPen(Qt::blue));
    customPlot->graph(0)->setLineStyle(QCPGraph::lsNone);
    customPlot->graph(0)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssCircle, 5));
    customPlot->graph(0)->setName("Intensity Samples");

    // Plot interpolated intensity (Rectangular)
    customPlot->addGraph();
    customPlot->graph(1)->setData(t_new, x_rect);
    customPlot->graph(1)->setPen(QPen(Qt::red, 2));
    customPlot->graph(1)->setName("Rectangular Window");

    // Plot interpolated intensity (Hamming)
    customPlot->addGraph();
    customPlot->graph(2)->setData(t_new, x_hamm);
    customPlot->graph(2)->setPen(QPen(Qt::green, 2));
    customPlot->graph(2)->setName("Hamming Window");

    // Plot interpolated intensity (Blackman)
    customPlot->addGraph();
    customPlot->graph(3)->setData(t_new, x_black);
    customPlot->graph(3)->setPen(QPen(Qt::magenta, 2));
    customPlot->graph(3)->setName("Blackman Window");

    // Configure plot
    customPlot->xAxis->setLabel("Position (nm)");
    customPlot->yAxis->setLabel("Intensity");
    customPlot->xAxis->setRange(0, 500);
    customPlot->yAxis->setRange(0, 1.2); // Normalized intensity
    customPlot->legend->setVisible(true);
    customPlot->legend->setFont(QFont("Helvetica", 9));
    customPlot->legend->setBrush(QBrush(QColor(255, 255, 255, 230)));
    customPlot->legend->setBorderPen(QPen(Qt::black));
    customPlot->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom);
    customPlot->replot();

    // Compute and print interpolation at x = 37 nm
    QVector<double> single_point = {37.0};
    double x_rect_37 = sincInterp(x, t, single_point, M, "rectangular")[0];
    double x_hamm_37 = sincInterp(x, t, single_point, M, "hamming")[0];
    double x_black_37 = sincInterp(x, t, single_point, M, "blackman")[0];
    std::cout << "Interpolated intensity at x = 37 nm (Rectangular): " << x_rect_37 << std::endl;
    std::cout << "Interpolated intensity at x = 37 nm (Hamming): " << x_hamm_37 << std::endl;
    std::cout << "Interpolated intensity at x = 37 nm (Blackman): " << x_black_37 << std::endl;

    window.show();
    return app.exec();
}