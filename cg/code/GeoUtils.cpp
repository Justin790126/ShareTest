#include "GeoUtils.h"

namespace jmk {
double areaTriangle2d(const Point2d &a, const Point2d &b, const Point2d &c) {
  auto AB = b - a;
  auto AC = c - a;
  auto result = crossProduct2D(AB, AC);
  return result / 2.0;
}

int orientation2d(const Point2d &a, const Point2d &b, const Point2d &c) {
  // +: left relative to ab
  // -: right relative to ab
  // 0: collinear
  auto area = areaTriangle2d(a, b, c);
  if (area > 0 && area < TOLERANCE) {
    area = 0;
  }

  if (area < 0 && area > TOLERANCE) {
    area = 0;
  }

  Vector2f ab = b - a;
  Vector2f ac = c - a;

  if (area > 0) {
    return (int)RELATIVE_POSITION::LEFT;
  }
  if (area < 0) {
    return (int)RELATIVE_POSITION::RIGHT;
  }

  if (ab[X] * ac[X] < 0 || ab[Y] * ac[Y] < 0) {
    return (int)RELATIVE_POSITION::BEHIND;
  }

  if (ab.magnitude() < ac.magnitude()) {
    return (int)RELATIVE_POSITION::BEYOND;
  }

  if (a == c) {
    return (int)RELATIVE_POSITION::ORIGIN;
  }
  if (b == c) {
    return (int)RELATIVE_POSITION::DESTINATION;
  }
  return (int)RELATIVE_POSITION::BETWEEN;
}

bool collinear(const Vector3f &a, const Vector3f &b) {
  auto v1 = a[X] * b[Y] - a[Y] * b[X];
  auto v2 = a[Y] * b[Z] - a[Z] * b[Y];
  auto v3 = a[X] * b[Z] - a[Z] * b[X];
  return isEqualD(v1, ZERO) && isEqualD(v2, ZERO) && isEqualD(v3, ZERO);
}

bool collinear(const Point3d &a, const Point3d &b, const Point3d &c) {
  auto AB = b - a;
  auto AC = c - a;
  return collinear(AB, AC);
}

bool coplaner(const Point3d &a, const Point3d &b, const Point3d &c,
              const Point3d &d) {
  auto AB = b - a;
  auto AC = c - a;
  auto AD = d - a;
  return coplaner(AB, AC, AD);
}

bool coplaner(const Vector3f &a, const Vector3f &b, const Vector3f &c) {
  float value = scalerTripleProduct(a, b, c);
  return isEqualD(value, ZERO);
}

bool left(const Point2d& a, const Point2d& b, const Point2d& c)
{
	return orientation2d(a, b, c) == (int)jmk::RELATIVE_POSITION::LEFT;
}
bool leftOrBeyond(const Point2d& a, const Point2d& b, const Point2d& c)
{
	int position = orientation2d(a, b, c);
	return (position == (int)RELATIVE_POSITION::LEFT || position == (int)RELATIVE_POSITION::BEYOND);
}

/*

This function checks whether the diagonal connecting two non-adjacent vertices
 v1 and v2 of a simple polygon lies entirely inside the polygon. 
 It’s likely used in algorithms like ear clipping for polygon triangulation,
  where valid diagonals must be inside the polygon to form triangles.


*/

static bool interiorCheck(const Vertex2d* v1, const Vertex2d* v2)
{
    // convex v1 < 180degree
    if (jmk::leftOrBeyond(v1->point, v1->next->point, v1->prev->point)) {
        return jmk::left(v1->point, v2->point, v1->prev->point) 
        && jmk::left(v2->point, v1->point, v1->next->point);
    }

    return !(jmk::leftOrBeyond(v1->point, v2->point, v1->next->point)
    && jmk::leftOrBeyond(v2->point, v1->point, v1->prev->point));
}

bool isDiagonal(const Vertex2d *v1, const Vertex2d *v2, PolygonS2d *poly) {
  bool prospect = true;
  std::vector<Vertex2d *> vertices;

  if (poly) {
    vertices = poly->getVertices();
  } else {
    auto vertex_ptr = v1->next;
    vertices.push_back((Vertex2d *)v1);
    while (vertex_ptr != v1) {
      vertices.push_back(vertex_ptr);
      vertex_ptr = vertex_ptr->next;
    }
  }

  Vertex2d *current, *next;
  current = vertices[0];
  do {
    next = current->next;
    if (current != v1 && next != v1 && current != v2 && next != v2 &&
        jmk::Intersection(v1->point, v2->point, current->point, next->point)) {
      prospect = false;
      break;
    }
    current = next;
  } while (current != vertices[0]);

  return prospect && interiorCheck(v1, v2) && interiorCheck(v2, v1);
}

} // namespace jmk
