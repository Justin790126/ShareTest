#ifndef ANGLE_H
#define ANGLE_H

#include "Line.h"
#include "Plane.h"

namespace jmk
{
    float AngleLines2D(const Line2d& l1, const Line2d& l2);
    float AngleLines3D(const Line3d& l1, const Line3d& L2);
}

#endif /* ANGLE_H */