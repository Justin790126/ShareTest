#include <vector>
#include <cmath>

// Compute sinc(x) = sin(pi * x) / (pi * x)
inline double sinc(double x) {
    if (x == 0.0) return 1.0;
    double pix = M_PI * x;
    return sin(pix) / pix;
}

// 2D windowed sinc interpolation
std::vector<double> sinc_interp_2d(
    const std::vector<std::vector<double>>& signal,
    const std::vector<double>& x_interp,
    const std::vector<double>& y_interp,
    int window_size = 8
) {
    size_t height = signal.size();           // Rows (y)
    size_t width = signal[0].size();         // Columns (x)
    size_t n_points = x_interp.size();

    std::vector<double> interpolated(n_points, 0.0);

    for (size_t i = 0; i < n_points; ++i) {
        double val = 0.0;
        for (size_t j = 0; j < height; ++j) {
            for (size_t k = 0; k < width; ++k) {
                double dx = x_interp[i] - static_cast<double>(k);
                double dy = y_interp[i] - static_cast<double>(j);

                if (std::abs(dx) <= window_size && std::abs(dy) <= window_size) {
                    // Blackman window in 2D
                    double wx = 0.42 + 0.5 * cos(M_PI * dx / window_size) + 0.08 * cos(2 * M_PI * dx / window_size);
                    double wy = 0.42 + 0.5 * cos(M_PI * dy / window_size) + 0.08 * cos(2 * M_PI * dy / window_size);
                    double sinc_x = sinc(dx);
                    double sinc_y = sinc(dy);
                    val += signal[j][k] * sinc_x * sinc_y * wx * wy;
                }
            }
        }
        interpolated[i] = val;
    }

    return interpolated;
}
