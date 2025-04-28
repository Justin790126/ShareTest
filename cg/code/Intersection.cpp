#include "Core.h"
#include "Intersection.h"
#include "GeoUtils.h"

bool jmk::Intersection(const jmk::Point2d & a, const jmk::Point2d & b,
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
  return _xor(ab_c==(int)RELATIVE_POSITION::LEFT, ab_d==(int)RELATIVE_POSITION::LEFT) && _xor(cd_a==(int)RELATIVE_POSITION::LEFT, cd_b==(int)RELATIVE_POSITION::LEFT);
}