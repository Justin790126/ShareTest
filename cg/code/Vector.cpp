#include "Vector.h"

namespace jmk {

float crossProduct2D(Vector2f v1, Vector2f v2)
{
    return v1[X] * v2[Y] - v1[Y] * v2[X];
}

Vector3f crossProduct3D(Vector3f v1, Vector3f v2)
{
    float x_, y_, z_;
    x_ = v1[Y] * v2[Z] - v1[Z] * v2[Y];
    y_ = v1[Z] * v2[X] - v1[X] * v2[Z];
    z_ = v1[X] * v2[Y] - v1[Y] * v2[X];
    return Vector3f(x_, y_, z_);
}

float scalerTripleProduct(Vector3f v1, Vector3f v2, Vector3f v3)
{
    auto bc_cross = crossProduct3D(v2, v3);
    return dotProduct(v1, bc_cross);
}

}