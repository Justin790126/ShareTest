#ifndef GEO_UTILS_H
#define GEO_UTILS_H

#include "Core.h"
#include "Point.h"
#include "Vector.h"
#include "Polygon.h"
#include "PolygonDCEL.h"
#include "Convexhull.h"
#include "Intersection.h"

namespace jmk
{
    
    double areaTriangle2d(const Point2d& a, const Point2d& b, const Point2d& c);

    int orientation2d(const Point2d& a, const Point2d& b, const Point2d& c);

    float FaceVisibility(const Face& _f, const Point3d& _p)
    {
        Point3d p1, p2, p3;

        p1 = *_f.vertices[0]->point;
        p2 = *_f.vertices[1]->point;
        p3 = *_f.vertices[2]->point;

        auto a1 = p2 - p1;
        auto b1 = p3 - p1;
        auto c1 = _p - p1;

        double vol1 = scalerTripleProduct(a1, b1, c1);
        return vol1;
    }

    bool collinear(const Vector3f& a, const Vector3f& b);

    bool collinear(const Point3d& a, const Point3d& b, const Point3d& c);

    bool coplaner(const Point3d& a, const Point3d& b, const Point3d& c, const Point3d& d);

    bool coplaner(const Vector3f& a, const Vector3f& b, const Vector3f& c);

    bool left(const Point2d& a, const Point2d& b, const Point2d& c);
    bool leftOrBeyond(const Point2d& a, const Point2d& b, const Point2d& c);

    bool isDiagonal(const Vertex2d* v1, const Vertex2d* v2, PolygonS2d* poly=NULL);

} // namespace jmk


#endif /* GEO_UTILS_H */