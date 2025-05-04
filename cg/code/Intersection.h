#ifndef INTERSECTION_H
#define INTERSECTION_H

#include "Line.h"
#include "Point.h"
#include "Plane.h"

namespace jmk {
bool Intersection(const jmk::Point2d &, const jmk::Point2d &,
                  const jmk::Point2d &, const jmk::Point2d &);
bool Intersection(const jmk::Point2d &, const jmk::Point2d &,
                  const jmk::Point2d &, const jmk::Point2d &, jmk::Point2d &);
bool Intersection(const jmk::Line2d &, const jmk::Line2d &, jmk::Point2d &);
bool Intersection(const jmk::Line3d &, const jmk::Planef &, jmk::Point3d &);
bool intersect(const jmk::Planef& ,jmk::Planef&, jmk::Line3d&);
} // namespace jmk

#endif /* INTERSECTION_H */