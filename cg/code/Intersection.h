#ifndef INTERSECTION_H
#define INTERSECTION_H

#include "Line.h"
#include "Point.h"

namespace jmk {
bool Intersection(const jmk::Point2d &, const jmk::Point2d &,
                  const jmk::Point2d &, const jmk::Point2d &);
bool Intersection(const jmk::Point2d &, const jmk::Point2d &,
                  const jmk::Point2d &, const jmk::Point2d &,
                  const jmk::Point2d &);
bool Intersection(const jmk::Line2d &, const jmk::Line2d &,
                  const jmk::Point2d &);
} // namespace jmk

#endif /* INTERSECTION_H */