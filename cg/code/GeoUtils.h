#ifndef GEO_UTILS_H
#define GEO_UTILS_H

#include "Core.h"
#include "Point.h"

namespace jmk
{
    
    double areaTriangle2d(const Point2d& a, const Point2d& b, const Point2d& c);

    int orientation2d(const Point2d& a, const Point2d& b, const Point2d& c);

    bool collinear(const Vector3f& a, const Vector3f& b);

    bool collinear(const Point3d& a, const Point3d& b, const Point3d& c);

    bool coplaner(const Point3d& a, const Point3d& b, const Point3d& c, const Point3d& d);

    bool coplaner(const Vector3f& a, const Vector3f& b, const Vector3f& c);

} // namespace jmk


#endif /* GEO_UTILS_H */