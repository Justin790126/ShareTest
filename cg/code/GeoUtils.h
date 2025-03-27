#ifndef GEO_UTILS_H
#define GEO_UTILS_H

#include "Core.h"
#include "Point.h"

namespace jmk
{
    
    double areaTriangle2d(const Point2d& a, const Point2d& b, const Point2d& c);

    int orientation2d(const Point2d& a, const Point2d& b, const Point2d& c);

} // namespace jmk


#endif /* GEO_UTILS_H */