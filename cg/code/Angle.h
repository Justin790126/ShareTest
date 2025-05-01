#ifndef ANGLE_H
#define ANGLE_H

#include "Line.h"
#include "Plane.h"

namespace jmk
{

    template<class T, size_t dim>
    float getAngle(jmk::Vector<T, dim>& v1, jmk::Vector<T, dim>& v2);

    float AngleLines2D(const Line2d& l1, const Line2d& l2);
    float AngleLines3D(const Line3d& l1, const Line3d& L2);
    float AngleLinePlane(const Line3d& l1, const Planef& p);
    float AnglePlane(const Planef& p1, const Planef& p2);
}

#endif /* ANGLE_H */