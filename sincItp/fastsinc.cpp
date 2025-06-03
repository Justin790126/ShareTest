#pragma once
#include <vector>
#include <cmath>
#include <cassert>

// Define sinc if not available
inline double sinc(double x) {
    if (std::abs(x) < 1e-8) return 1.0;
    return std::sin(M_PI * x) / (M_PI * x);
}

// Table for sinc and Blackman window values, with linear interpolation
class SincBlackmanLookupTable {
public:
    SincBlackmanLookupTable(int window_size, int samples_per_unit)
        : window_size_(window_size),
          samples_per_unit_(samples_per_unit),
          min_x_(-window_size),
          max_x_(window_size)
    {
        table_size_ = (max_x_ - min_x_) * samples_per_unit_ + 1;
        sinc_table_.resize(table_size_);
        blackman_table_.resize(table_size_);
        step_ = 1.0 / samples_per_unit_;
        for (int i = 0; i < table_size_; ++i) {
            double dx = min_x_ + i * step_;
            sinc_table_[i] = sinc(dx);
            blackman_table_[i] = 0.42 + 0.5 * std::cos(M_PI * dx / window_size_) + 0.08 * std::cos(2 * M_PI * dx / window_size_);
        }
    }

    // Linear interpolation lookup
    double sinc(double x) const {
        if (x < min_x_ || x > max_x_) return 0.0;
        double pos = (x - min_x_) / step_;
        int idx = static_cast<int>(pos);
        double frac = pos - idx;
        if (idx < 0) return sinc_table_.front();
        if (idx + 1 >= table_size_) return sinc_table_.back();
        return sinc_table_[idx] * (1.0 - frac) + sinc_table_[idx + 1] * frac;
    }
    double blackman(double x) const {
        if (x < min_x_ || x > max_x_) return 0.0;
        double pos = (x - min_x_) / step_;
        int idx = static_cast<int>(pos);
        double frac = pos - idx;
        if (idx < 0) return blackman_table_.front();
        if (idx + 1 >= table_size_) return blackman_table_.back();
        return blackman_table_[idx] * (1.0 - frac) + blackman_table_[idx + 1] * frac;
    }

    int window_size() const { return window_size_; }

private:
    int window_size_;
    int samples_per_unit_;
    int table_size_;
    double min_x_, max_x_, step_;
    std::vector<double> sinc_table_;
    std::vector<double> blackman_table_;
};

// Interpolation function using the lookup table
inline std::vector<double> sinc_interp_2d_with_lookup_table(
    const std::vector<std::vector<double>>& signal,
    const std::vector<double>& x_interp,
    const std::vector<double>& y_interp,
    const SincBlackmanLookupTable& table
) {
    int window_size = table.window_size();
    size_t height = signal.size();
    assert(height > 0);
    size_t width = signal[0].size();
    assert(width > 0);
    size_t n_points = x_interp.size();
    assert(y_interp.size() == n_points);

    std::vector<double> interpolated(n_points, 0.0);

    for (size_t i = 0; i < n_points; ++i) {
        double x = x_interp[i];
        double y = y_interp[i];

        int x0 = static_cast<int>(std::floor(x));
        int y0 = static_cast<int>(std::floor(y));

        double val = 0.0;
        for (int j = y0 - window_size; j <= y0 + window_size; ++j) {
            if (j < 0 || j >= static_cast<int>(height)) continue;
            double dy = y - j;
            double sinc_y = table.sinc(dy);
            double wy = table.blackman(dy);

            for (int k = x0 - window_size; k <= x0 + window_size; ++k) {
                if (k < 0 || k >= static_cast<int>(width)) continue;
                double dx = x - k;
                double sinc_x = table.sinc(dx);
                double wx = table.blackman(dx);

                val += signal[j][k] * sinc_x * sinc_y * wx * wy;
            }
        }
        interpolated[i] = val;
    }
    return interpolated;
}

int main()
{
    SincBlackmanLookupTable table(4, 100);
// Call the function:
std::vector<double> result = sinc_interp_2d_with_lookup_table(signal, x_interp, y_interp, table);
return 0;
}