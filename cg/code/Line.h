#ifndef LINE_H
#define LINE_H

#include "Vector.h"

namespace jmk {

    template<class coord_type, size_t dim=DIM3>
    class Line {
        Vector<coord_type, dim> point;
        Vector<coord_type, dim> dir;

        public:
            Line() {}

            Line(Vector<coord_type, dim>& p1, Vector<coord_type, dim>& p2) {
                dir = p2 - p1;
                point = p1;
            }

            Vector<coord_type, dim> getPoint();

            Vector<coord_type, dim> getDir();

    };

    template<class coord_type, size_t dim>
    inline Vector<coord_type, dim> Line<coord_type, dim>::getPoint()
    {
        return point;
    }

    template<class coord_type, size_t dim>
    inline Vector<coord_type, dim> Line<coord_type, dim>::getDir()
    {
        return dir;
    }


}


#endif /* LINE_H */