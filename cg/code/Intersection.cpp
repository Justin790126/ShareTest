#include "Intersection.h"
#include "Core.h"
#include "GeoUtils.h"

bool jmk::Intersection(const jmk::Point2d &a, const jmk::Point2d &b,
                       const jmk::Point2d &c, const jmk::Point2d &d) {
  auto ab_c = jmk::orientation2d(a, b, c);
  auto ab_d = jmk::orientation2d(a, b, d);
  auto cd_a = jmk::orientation2d(c, d, a);
  auto cd_b = jmk::orientation2d(c, d, b);
  if (ab_d == (int)RELATIVE_POSITION::BETWEEN ||
      ab_d == (int)RELATIVE_POSITION::ORIGIN ||
      ab_d == (int)RELATIVE_POSITION::DESTINATION ||
      ab_c == (int)RELATIVE_POSITION::BETWEEN ||
      ab_c == (int)RELATIVE_POSITION::ORIGIN ||
      ab_c == (int)RELATIVE_POSITION::DESTINATION ||
      cd_a == (int)RELATIVE_POSITION::BETWEEN ||
      cd_a == (int)RELATIVE_POSITION::ORIGIN ||
      cd_a == (int)RELATIVE_POSITION::DESTINATION ||
      cd_b == (int)RELATIVE_POSITION::BETWEEN ||
      cd_b == (int)RELATIVE_POSITION::ORIGIN ||
      cd_b == (int)RELATIVE_POSITION::DESTINATION) {
    return true;
  }
  return _xor(ab_c == (int)RELATIVE_POSITION::LEFT,
              ab_d == (int)RELATIVE_POSITION::LEFT) &&
         _xor(cd_a == (int)RELATIVE_POSITION::LEFT,
              cd_b == (int)RELATIVE_POSITION::LEFT);
}

bool jmk::Intersection(const jmk::Point2d &a, const jmk::Point2d &b,
                       const jmk::Point2d &c, const jmk::Point2d &d,
                       jmk::Point2d &intersection) {
  Vector2f AB = b - a;
  Vector2f CD = d - c;

  Vector2f n(CD[Y], -CD[X]);

  auto deno = dotProduct(n, AB);

  if (!isEqualD(deno, ZERO)) {
    auto AC = c - a;
    auto numer = dotProduct(n, AC);

    auto t = numer / deno;
    float x = a[X] + t * AB[X];
    float y = a[Y] + t * AB[Y];

    intersection.assign(X, x);
    intersection.assign(Y, y);

    return true;
  } else {
    // parallel
    return false;
  }
}

bool jmk::Intersection(const jmk::Line2d &l1, const jmk::Line2d &l2,
                       jmk::Point2d &intersection) {

  auto l1s = l1.getPoint();
  auto l1e = l1s + l1.getDir();
  auto l2s = l2.getPoint();
  auto l2e = l2s + l2.getDir();

  return Intersection(l1s, l1e, l2s, l2e, intersection);
}

bool jmk::Intersection(const jmk::Line3d &line, const jmk::Planef &plane,
                       jmk::Point3d &point) {
  auto n = plane.getNormal();
  auto D = plane.getD();
  auto d = line.getDir();
  auto p = line.getPoint();
  auto  nd = dotProduct(n,d);
  auto dot = dotProduct(n, d);
  if (!isEqualD(dot, ZERO)) {
    auto t = (-1*dotProduct(n, p) + D)/nd;
    point.assign(X, p[X]+t*d[X]);
    point.assign(Y, p[Y]+t*d[Y]);
    point.assign(Z, p[Z]+t*d[Z]);
    return true;
  } else {
    return false;
  }
}

bool jmk::intersect(const jmk::Planef& p1 ,jmk::Planef& p2, jmk::Line3d& l)
{
  auto n1 = p1.getNormal();
  auto n2 = p2.getNormal();
  auto d1 = p1.getD();
  auto d2 = p2.getD();

  auto direction = crossProduct3D(n1, n2);

  if (isEqualD(direction.magnitude(), ZERO)) {
    return false;
  }

  auto n1n2 = dotProduct(n1, n2);
  auto n1n2_2 = n1n2 * n1n2;

  auto a = (d2*n1n2-d1)/(n1n2_2 -1);
  auto b = (d1*n1n2-d2)/(n1n2_2 -1);

  auto point = n1*a + n2*b;

  l.setPoint(point);
  direction.normalize();
  l.setDirection(direction);

  return true;
}