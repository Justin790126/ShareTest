#include "Angle.h"


float jmk::AngleLines2D(const Line2d& l1, const Line2d& l2)
{
    auto l1_mag = l1.getDir().magnitude();
    auto l2_mag = l2.getDir().magnitude();

    auto dot = dotProduct(l1.getDir(), l2.getDir());

    auto theta = acos(fabs(dot)/(l1_mag*l2_mag));
    return RadianceToDegrees(theta);
}

float jmk::AngleLines3D(const Line3d& l1, const Line3d& l2)
{
    auto l1_mag = l1.getDir().magnitude();
    auto l2_mag = l2.getDir().magnitude();

    auto dot = dotProduct(l1.getDir(), l2.getDir());

    auto theta = acos(fabs(dot)/(l1_mag*l2_mag));
    return RadianceToDegrees(theta);
}