#include "precomputed_interpolator2d.h"

static double blackman_window(double x, int M) {
    double n = x + M / 2.0;
    if (n < 0 || n > M) return 0.0;
    return 0.42 - 0.5 * cos(2 * PI * n / M) + 0.08 * cos(4 * PI * n / M);
}

static double sinc(double x) {
    if (fabs(x) < 1e-8) return 1.0;
    return sin(PI * x) / (PI * x);
}

PrecomputedInterpolator2D::PrecomputedInterpolator2D(
    const std::vector<std::vector<double>>& img,
    const std::vector<double>& x_pix,
    const std::vector<double>& y_pix,
    int window,
    int n_sample // NEW: number of samples per interpolation point
) : img_(img), n_sample_(n_sample)
{
    imgY_ = img.size();
    imgX_ = imgY_ ? img[0].size() : 0;
    coeffs_.resize(x_pix.size());
    for (size_t idx = 0; idx < x_pix.size(); ++idx) {
        int x0 = floor(x_pix[idx]);
        int y0 = floor(y_pix[idx]);
        std::vector<int> dxs, dys;
        std::vector<double> ws;
        double norm = 0.0;
        for (int j = -window; j <= window; ++j) {
            int yj = y0 + j;
            if (yj < 0 || yj >= imgY_) continue;
            double wy = sinc(y_pix[idx] - yj) * blackman_window(y_pix[idx] - yj, 2 * window + 1);
            for (int i = -window; i <= window; ++i) {
                int xi = x0 + i;
                if (xi < 0 || xi >= imgX_) continue;
                double wx = sinc(x_pix[idx] - xi) * blackman_window(x_pix[idx] - xi, 2 * window + 1);
                double w = wx * wy;
                dxs.push_back(xi);
                dys.push_back(yj);
                ws.push_back(w);
                norm += w;
            }
        }
        // Normalize
        if (norm == 0.0) norm = 1.0;
        for (double& w : ws) w /= norm;
        coeffs_[idx] = InterpCoeff2D{dxs, dys, ws};
    }
}

double PrecomputedInterpolator2D::interpolate(size_t i) const {
    double sum = 0.0;
    const auto& c = coeffs_[i];
    for (size_t j = 0; j < c.weight.size(); ++j) {
        sum += img_[c.dy[j]][c.dx[j]] * c.weight[j];
    }
    return sum;
}