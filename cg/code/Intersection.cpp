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
                       const jmk::Point2d &intersection) {
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
                       const jmk::Point2d &intersection) {

  auto l1s = l1.getPoint();
  auto l1e = l1s + l1.getDir();
  auto l2s = l2.getPoint();
  auto l2e = l2s + l2.getDir();

  return Intersection(l1s, l1e, l2s, l2e, intersection);
}