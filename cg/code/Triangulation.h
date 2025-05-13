#ifndef TRIANGULATION_H
#define TRIANGULATION_H

#include "Polygon.h"
#include "GeoUtils.h"

namespace jmk
{
    void Triangulation_earclipping(PolygonS2d* poly, std::vector<jmk::Edge2d>& edge_list);
}

#endif /* TRIANGULATION_H */