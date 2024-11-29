#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

double cubicSplineInterpolation(const std::vector<double>& x, const std::vector<double>& y, double xq) {
    int n = x.size();
    std::vector<double> h(n - 1), alpha(n - 1), l(n), mu(n - 1), z(n);

    // Calculate h, alpha
    for (int i = 0; i < n - 1; ++i) {
        h[i] = x[i + 1] - x[i];
        alpha[i] = (3.0 / h[i]) * (y[i + 1] - y[i]) - (3.0 / h[i - 1]) * (y[i] - y[i - 1]);
    }

    // Calculate l, mu, z
    l[0] = 2 * h[0];
    mu[0] = 0.5;
    z[0] = alpha[0] / l[0];
    for (int i = 1; i < n - 1; ++i) {
        l[i] = 2 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1];
        mu[i] = h[i] / l[i];
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
    }

    // Calculate coefficients
    std::vector<double> c(n), b(n), d(n - 1);
    c[n - 1] = 0;
    for (int i = n - 2; i >= 0; --i) {
        c[i] = z[i] - mu[i] * c[i + 1];
        b[i] = (y[i + 1] - y[i]) / h[i] - h[i] * (c[i + 1] + 2 * c[i]) / 3.0;
        d[i] = (c[i + 1] - c[i]) / (3.0 * h[i]);
    }

    // Find the interval containing xq
    int i = 0;
    while (i < n - 1 && xq > x[i + 1]) {
        ++i;
    }

    // Interpolate using the cubic spline formula
    double dx = xq - x[i];
    return y[i] + dx * (b[i] + dx * (c[i] + dx * d[i]));
}

double interpolateZ(const std::vector<double>& x, const std::vector<double>& y, const std::vector<double>& z, double xq, double yq) {
    // Grid the data
    std::vector<double> x_grid, y_grid;
    for (double x_val : x) {
        x_grid.push_back(x_val);
    }
    for (double y_val : y) {
        y_grid.push_back(y_val);
    }

    std::sort(x_grid.begin(), x_grid.end());
    std::sort(y_grid.begin(), y_grid.end());

    // Interpolate z-values for each grid point
    std::vector<std::vector<double>> z_grid(x_grid.size(), std::vector<double>(y_grid.size()));
    for (int i = 0; i < x_grid.size(); ++i) {
        for (int j = 0; j < y_grid.size(); ++j) {
            std::vector<double> x_vals, z_vals;
            for (int k = 0; k < x.size(); ++k) {
                if (std::abs(y[k] - y_grid[j]) < 1e-6) {
                    x_vals.push_back(x[k]);
                    z_vals.push_back(z[k]);
                }
            }
            z_grid[i][j] = cubicSplineInterpolation(x_vals, z_vals, x_grid[i]);
        }
    }

    // Find the four nearest grid points to (xq, yq)
    int i = 0, j = 0;
    while (i < x_grid.size() - 1 && xq > x_grid[i + 1]) {
        ++i;
    }
    while (j < y_grid.size() - 1 && yq > y_grid[j + 1]) {
        ++j;
    }

    double x1 = x_grid[i], x2 = x_grid[i + 1];
    double y1 = y_grid[j], y2 = y_grid[j + 1];

    // Bilinear interpolation to estimate zq
    double z11 = z_grid[i][j];
    double z12 = z_grid[i][j + 1];
    double z21 = z_grid[i + 1][j];
    double z22 = z_grid[i + 1][j + 1];

    double dx = (xq - x1) / (x2 - x1);
    double dy = (yq - y1) / (y2 - y1);

    double zq = (1 - dx) * (1 - dy) * z11 + dx * (1 - dy) * z21 + (1 - dx) * dy * z12 + dx * dy * z22;

    return zq;
}

int main() {
    // Sample data
    std::vector<double> x = {1, 2, 3, 4, 5};
    std::vector<double> y = {1, 2, 3, 4, 5};
    std::vector<double> z = {2, 3, 4, 5, 6};

    // Given (xq, yq)
    double xq = 2.5;
    double yq = 3.2;

    double zq = interpolateZ(x, y, z, xq, yq);

    std::cout << "Interpolated z-value at (" << xq << ", " << yq << ") is: " << zq << std::endl;

    return 0;
}