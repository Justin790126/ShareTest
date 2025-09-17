#include "Angle.h"
#include "Point.h"
#include <list>
#include <vector>

using namespace std;

namespace jmk {

float polarAngle(const Point2d &point, const Point2d &reference) {
  // Compute the vector from reference to point
  Vector2f vec = point - reference;

  // Calculate the polar angle using atan2
  // atan2 returns angle in radians between -π and π
  float angle_rad = std::atan2(vec[Y], vec[X]);

  // Convert radians to degrees
  float angle_deg = angle_rad * 180.0f / M_PI;

  // Ensure the angle is in [0, 360) range
  if (angle_deg < 0) {
    angle_deg += 360.0f;
  }

  return angle_deg;
}

void convexhull2DGiftwrapping(vector<Point2d> &_points,
                              vector<Point2d> &_convex) {
  if (_points.size() <= 3) {
    return;
  }

  Point2d bottom_point = _points[0];

  for (Point2d &point : _points) {
    if ((point[Y] < bottom_point[Y]) ||
        (point[Y] == bottom_point[Y] && (point[X] < bottom_point[X]))) {
      bottom_point = point;
    }
  }

  Point2d min_polar_point = _points[0];
  float current_polor_angle = 360;

  for (size_t i = 0; i < _points.size(); i++) {
    float polar_angle = polarAngle(_points[i], bottom_point);
    if (bottom_point != _points[i] && current_polor_angle > polar_angle) {
      current_polor_angle = polar_angle;
      min_polar_point = _points[i];
    }
  }

  // add the first two points of the convexhull
  _convex.push_back(bottom_point);
  _convex.push_back(min_polar_point);

  Point2d ref_point = min_polar_point;
  int index_before_last = 0;
  while (true) {
    current_polor_angle = 360;
    for (size_t i = 0; i < _points.size(); i++) {
      Vector2f vec1 = ref_point - _convex[index_before_last];
      Vector2f vec2 = _points[i] - ref_point;

      float between_angle = getAngle(vec1, vec2);
      if (ref_point != _points[i] && current_polor_angle > between_angle) {
        current_polor_angle = between_angle;
        min_polar_point = _points[i];
      }

      if (min_polar_point == bottom_point) {
        break;
      }

      index_before_last++;
      _convex.push_back(min_polar_point);
      ref_point = min_polar_point;
    }
  }
}

} // namespace jmk