#include "GeoUtils.h"

namespace jmk
{
    double areaTriangle2d(const Point2d& a, const Point2d& b, const Point2d& c)
    {
        auto AB = b -a;
        auto AC = c -a;
        auto result = crossProduct2D(AB, AC);
        return result / 2.0;
    }

    int orientation2d(const Point2d& a, const Point2d& b, const Point2d& c)
    {
        // +: left relative to ab
        // -: right relative to ab
        // 0: collinear
        auto area = areaTriangle2d(a, b, c);
        if (area > 0 && area < TOLERANCE) {
            area = 0;
        }

        if (area < 0 && area > TOLERANCE) {
            area = 0;
        }

        Vector2f ab = b-a;
        Vector2f ac = c-a;

        if (area > 0) {
            return (int)RELATIVE_POSITION::LEFT;
        }
        if (area < 0) {
            return (int)RELATIVE_POSITION::RIGHT;
        }

        if (ab[X]*ac[X] < 0 || ab[Y]*ac[Y] < 0) {
            return (int)RELATIVE_POSITION::BEHIND;
        }

        if (ab.magnitude() < ac.magnitude()) {
            return (int)RELATIVE_POSITION::BEYOND;
        }

        if (a == c) {
            return (int)RELATIVE_POSITION::ORIGIN;
        }
        if (b == c) {
            return (int)RELATIVE_POSITION::DESTINATION;
        }
        return (int)RELATIVE_POSITION::BETWEEN;
    }
}

