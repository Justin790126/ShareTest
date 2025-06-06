#pragma once
#include <vector>
#include <map>
#include <cmath>
#define PI 3.14159265358979323846

struct InterpCoeff2D {
    // Offsets and weights relative to (x0, y0)
    std::vector<int> dx, dy;
    std::vector<double> weight;
};

class PrecomputedInterpolator2D {
public:
    PrecomputedInterpolator2D(
        const std::vector<std::vector<double>>& img,
        const std::vector<double>& x_pix,
        const std::vector<double>& y_pix,
        int window = 4,
        int n_sample = 10 // NEW: number of samples per interpolation point
    );

    // Interpolated value at precomputed point i
    double interpolate(size_t i) const;

    // Optionally: get the number of samples per point
    int num_samples() const { return n_sample_; }

private:
    const std::vector<std::vector<double>>& img_;
    std::vector<InterpCoeff2D> coeffs_;
    int imgX_, imgY_;
    int n_sample_; // NEW: store number of samples
};