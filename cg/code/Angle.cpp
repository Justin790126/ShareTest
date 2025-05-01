#include "Angle.h"

template<class T, size_t dim>
float jmk::getAngle(jmk::Vector<T, dim>& v1, jmk::Vector<T, dim>& v2)
{
    auto dot = dotProduct(v1, v2);
    auto theta = acos(fabs(dot));
    return RadianceToDegrees(theta);
}


float jmk::AngleLines2D(const Line2d& l1, const Line2d& l2)
{
    Vector<float, DIM2> v1 = l1.getDir();
    Vector<float, DIM2> v2 = l2.getDir();
    return getAngle(v1, v2);
}

float jmk::AngleLines3D(const Line3d& l1, const Line3d& l2)
{
    Vector<float, DIM3> v1 = l1.getDir();
    Vector<float, DIM3> v2 = l2.getDir();
    return getAngle(v1, v2);
}

float jmk::AngleLinePlane(const Line3d& l1, const Planef& p)
{
    Vector<float, DIM3> v1 = l1.getDir();
    Vector<float, DIM3> v2 = p.getNormal();
    auto angle = getAngle(v1, v2);
    return 90-angle;
}

float jmk::AnglePlane(const Planef& p1, const Planef& p2)
{
    Vector<float, DIM3> v1 = p1.getNormal();
    Vector<float, DIM3> v2 = p2.getNormal();
    return getAngle(v1, v2);
}