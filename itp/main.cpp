#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <iomanip>
#include <algorithm>

#define PI 3.14159265358979323846

using namespace std;

#include "data.h"
#include "qcustomplot.h"
#include <QApplication> // Use QApplication, not QtGui

extern vector<double> x_slice_point_nm;
extern vector<double> y_slice_point_nm;
extern vector<vector<double>> img;

// Blackman window function
double blackman_window(double x, int M) {
    // Blackman window: w[n] = 0.42 - 0.5*cos(2*pi*n/M) + 0.08*cos(4*pi*n/M)
    // x in [-M/2, M/2]
    double n = x + M / 2.0;
    if (n < 0 || n > M) return 0.0;
    double w = 0.42 - 0.5 * cos(2 * PI * n / M) + 0.08 * cos(4 * PI * n / M);
    return w;
}

double sinc(double x) {
    if (fabs(x) < 1e-8) return 1.0;
    return sin(PI * x) / (PI * x);
}

// 2D Blackman-windowed sinc interpolation
double interpolate2D(const vector<vector<double>>& img,
                     double x, double y, int window =4) {
    int imgY = img.size();
    int imgX = imgY ? img[0].size() : 0;
    int x0 = floor(x);
    int y0 = floor(y);

    double result = 0.0;
    double norm = 0.0;
    for (int j = -window; j <= window; ++j) {
        int yj = y0 + j;
        if (yj < 0 || yj >= imgY) continue;
        double wy = sinc(y - yj) * blackman_window(y - yj, 2 * window + 1);
        for (int i = -window; i <= window; ++i) {
            int xi = x0 + i;
            if (xi < 0 || xi >= imgX) continue;
            double wx = sinc(x - xi) * blackman_window(x - xi, 2 * window + 1);
            double w = wx * wy;
            result += img[yj][xi] * w;
            norm += w;
        }
    }
    if (norm == 0.0) return 0.0;
    return result / norm;
}

int main(int argc, char *argv[]) {

    QApplication app(argc, argv);

    // Image dimensions
    int pixX = 70;
    int pixY = 70;

    // Use image data as is (assume extern is populated)
    vector<vector<double>> img2 = img;

    // Interpolation parameters
    double bottomLeftX = 1418497.0;
    double bottomLeftY = -3339036;
    double simNmPerPix = 4.0;

    // Interpolated values
    vector<double> interpolated_values;

    // Interpolate at each (x_nm, y_nm)
    for (size_t i = 0; i < x_slice_point_nm.size() && i < y_slice_point_nm.size(); ++i) {
        double x_nm = x_slice_point_nm[i];
        double y_nm = y_slice_point_nm[i];
        // Convert nanometers to pixel coordinates
        double x_pix = (x_nm - bottomLeftX) / simNmPerPix;
        double y_pix = (y_nm - bottomLeftY) / simNmPerPix;
        double val = interpolate2D(img2, x_pix, y_pix);
        printf("%f, ", val);
        interpolated_values.push_back(val);
    }
    printf("\n");

    // use qcustomplot to plot distance_nm in x, interpolated_values in y
    QCustomPlot customPlot;
    QVector<double> xData, yData;
    for (size_t i = 0; i < x_slice_point_nm.size() && i < interpolated_values.size(); i++) {
        xData.append(x_slice_point_nm[i]);
        yData.append(interpolated_values[i]);
    }
    customPlot.addGraph();
    customPlot.graph(0)->setData(xData, yData);
    // add scatter style to disk
    customPlot.graph(0)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssDisc, 5));
    customPlot.xAxis->setLabel("Distance (nm)");
    customPlot.yAxis->setLabel("Interpolated Value");
    if (!x_slice_point_nm.empty() && !interpolated_values.empty()) {
        customPlot.xAxis->setRange(
            *min_element(x_slice_point_nm.begin(), x_slice_point_nm.end()),
            *max_element(x_slice_point_nm.begin(), x_slice_point_nm.end()));
        customPlot.yAxis->setRange(
            *min_element(interpolated_values.begin(), interpolated_values.end()),
            *max_element(interpolated_values.begin(), interpolated_values.end()));
    }
    customPlot.replot();
    customPlot.resize(800, 600);
    customPlot.show();

    return app.exec();
}