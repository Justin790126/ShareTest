#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

#include "data.h"
#include "qcustomplot.h"
#include <QApplication>
#include "precomputed_interpolator2d.h"

extern std::vector<double> x_slice_point_nm;
extern std::vector<double> y_slice_point_nm;
extern std::vector<std::vector<double>> img;

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);

    // Params
    double bottomLeftX = 1418497.0;
    double bottomLeftY = -3339036;
    double simNmPerPix = 4.0;

    // Convert sample points to pixel coordinates
    std::vector<double> x_pix, y_pix;
    for (size_t i = 0; i < x_slice_point_nm.size() && i < y_slice_point_nm.size(); ++i) {
        x_pix.push_back((x_slice_point_nm[i] - bottomLeftX) / simNmPerPix);
        y_pix.push_back((y_slice_point_nm[i] - bottomLeftY) / simNmPerPix);
    }

    // New parameter: number of samples
    int window = 4;         // window size for interpolation kernel
    int n_sample = 10;       // set your desired number of samples per interpolation point

    // Precompute coefficients using new class constructor
    PrecomputedInterpolator2D interp(img, x_pix, y_pix, window, n_sample);

    // Compute interpolated values
    std::vector<double> interpolated_values(x_pix.size());
    for (size_t i = 0; i < x_pix.size(); ++i) {
        interpolated_values[i] = interp.interpolate(i);
        printf("%f, ", interpolated_values[i]);
    }

    // Plot
    QCustomPlot customPlot;
    QVector<double> xData, yData;
    for (size_t i = 0; i < x_slice_point_nm.size() && i < interpolated_values.size(); i++) {
        xData.append(x_slice_point_nm[i]);
        yData.append(interpolated_values[i]);
    }
    customPlot.addGraph();
    customPlot.graph(0)->setData(xData, yData);
    customPlot.xAxis->setLabel("Distance (nm)");
    customPlot.yAxis->setLabel("Interpolated Value");
    // add scatter style to disc
    customPlot.graph(0)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssDisc, 5));
    // set interaction
    customPlot.setInteractions(QCP::iRangeDrag | QCP::iRangeZoom | QCP::iSelectPlottables);
    if (!x_slice_point_nm.empty() && !interpolated_values.empty()) {
        customPlot.xAxis->setRange(
            *std::min_element(x_slice_point_nm.begin(), x_slice_point_nm.end()),
            *std::max_element(x_slice_point_nm.begin(), x_slice_point_nm.end()));
        customPlot.yAxis->setRange(
            *std::min_element(interpolated_values.begin(), interpolated_values.end()),
            *std::max_element(interpolated_values.begin(), interpolated_values.end()));
    }
    customPlot.replot();
    customPlot.resize(800, 600);
    customPlot.show();
    return app.exec();
}