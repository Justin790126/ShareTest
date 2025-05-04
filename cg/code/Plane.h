#ifndef PLANE_H
#define PLANE_H

#include "Vector.h"
#include "Point.h"

namespace jmk
{

    template<class coord_type>
    class Plane {
        Vector3f normal;
        float d = 0;

        /*
            Ax+By+Cz = d
            normal: (A, B, C)
            d: dot product of input point 
        */ 


        public:
            Plane() {}

            Plane(Vector3f& _normal, float _const) : normal(_normal), d(_const) {
                normal.normalize();
            }
            Plane(Point3d& _p1, Point3d& _p2, Point3d& _p3) {
                // cross product get normal vector
                auto v12 = _p2-_p1;
                auto v13 = _p3-_p1;

                normal = crossProduct3D(v12, v13);
                normal.normalize();
                d = dotProduct(normal, _p1);
            }

            Vector3f getNormal() const {
                return normal;
            }

            float getD() const {
                return d;
            }

    };

    typedef Plane<float> Planef;
};

#endif /* PLANE_H */