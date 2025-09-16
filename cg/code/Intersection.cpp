#include "Intersection.h"
#include "Core.h"
#include "GeoUtils.h"
#include <vector>
#include <algorithm>
#include <map>
#include <set>

using namespace std;

namespace jmk {

class Event {
  private:
   Point2d* point;

  public:
    Event() : point() {};
    Event(const Event& e) : point(e.point) {};
    Event(const Point2d* p) : point((Point2d*)p) {};

    bool operator==(const Event& _other) const {
        return (*point) == ((*((Event)_other).get_point()));
    }

    Point2d* get_point() const {
        return point;
    }
};

class Segment2d
{
  private:
    

  public:
    Segment2d() : p1(), p2() {}
    Segment2d(const Point2d& s, const Point2d& e) : p1(s), p2(e) {}
    Segment2d(const Segment2d& seg) : p1(seg.p1), p2(seg.p2) {}

    Point2d p1;
    Point2d p2;

    float get_x(const Point2d& ref) const
    {
      if (isEqualD(p1[Y], p2[Y])) {
        return min(p1[X], p2[X]);
      }

      float m = (p2[Y] - p1[Y]) / (p2[X] - p1[X]);
      float b = p1[Y] - m * p1[X];

      return (ref[Y] - b) / m;
    }

};

static bool is_left(Segment2d* _current, Segment2d* _other, Point2d* _ref)
{
  return _current->get_x(*_ref) < _other->get_x(*_ref);
}

struct CustomComparator {
  Point2d* sweep_line_point;
  CustomComparator(Point2d* p) : sweep_line_point(p) {}

  bool operator()(Segment2d* _a, Segment2d* _b) const {
    if (_a == _b) {
      return false;
    }
    return is_left(_a, _b, sweep_line_point);
  }
};

template<typename T>
struct EventComparator {
  bool operator()(const T& e1, const T& e2) const
  {
    Point2d* point = ((T&)e1).get_point();
    Point2d* other = ((T&)e2).get_point();

    if ((*point)[Y] > (*other)[Y]) {
      return true;
    } else if (isEqualD((*point)[Y], (*other)[Y]) && (*point)[X] < (*other)[X]) {
      return true;
    }

    return false;
  }
};

template<typename T>
class PriorityQueue {
  std::map<T, std::vector<Segment2d*>, EventComparator<T>> base_map;

public:
  void push(T& value) {
    std::vector<Segment2d*> v;
    base_map.insert(std::pair<T, std::vector<Segment2d*>>(value, v));
  }

  void push(T& value, Segment2d* seg)
  {
    std::vector<Segment2d*> v;
    v.push_back(seg);

    // Corrected line
    std::pair<typename std::map<T, std::vector<Segment2d*>, EventComparator<T>>::iterator, bool>
      ret = base_map.insert(std::pair<T, std::vector<Segment2d*>>(value, v));
    if (!ret.second) {
      ret.first->second.push_back(seg);
    }
  }

  T top_event() {
    return base_map.begin()->first;
  }

  std::vector<Segment2d*> top_seglist() {
    return base_map.begin()->second;
  }

  void pop() {
    base_map.erase(base_map.begin());
  }

  bool empty()
  {
    return base_map.empty();
  }

};

void intersections(std::list<Segment2d*>& segment_list)
{
  PriorityQueue<Event> queue;

  Point2d sweep_point(0.0, 0.0);
  Point2d previous(0.0, 0.0);
  CustomComparator comp(&sweep_point);
  std::set<Segment2d*, CustomComparator> sweep_line_status(comp);
  typedef std::set<Segment2d*, CustomComparator>::iterator sweep_line_itr;
  
  for (auto seg: segment_list) {
    Event top_event(&seg->p1);
    Event bot_event(&seg->p2);

    queue.push(top_event, seg);
    queue.push(bot_event);
  }
}

bool Intersection(const Point2d &a, const Point2d &b, const Point2d &c,
                  const Point2d &d) {
  auto ab_c = orientation2d(a, b, c);
  auto ab_d = orientation2d(a, b, d);
  auto cd_a = orientation2d(c, d, a);
  auto cd_b = orientation2d(c, d, b);
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

bool Intersection(const Point2d &a, const Point2d &b, const Point2d &c,
                  const Point2d &d, Point2d &intersection) {
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

bool Intersection(const Line2d &l1, const Line2d &l2, Point2d &intersection) {

  auto l1s = l1.getPoint();
  auto l1e = l1s + l1.getDir();
  auto l2s = l2.getPoint();
  auto l2e = l2s + l2.getDir();

  return Intersection(l1s, l1e, l2s, l2e, intersection);
}

bool Intersection(const Line3d &line, const Planef &plane, Point3d &point) {
  auto n = plane.getNormal();
  auto D = plane.getD();
  auto d = line.getDir();
  auto p = line.getPoint();
  auto nd = dotProduct(n, d);
  auto dot = dotProduct(n, d);
  if (!isEqualD(dot, ZERO)) {
    auto t = (-1 * dotProduct(n, p) + D) / nd;
    point.assign(X, p[X] + t * d[X]);
    point.assign(Y, p[Y] + t * d[Y]);
    point.assign(Z, p[Z] + t * d[Z]);
    return true;
  } else {
    return false;
  }
}

bool intersect(const Planef &p1, Planef &p2, Line3d &l) {
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

  auto a = (d2 * n1n2 - d1) / (n1n2_2 - 1);
  auto b = (d1 * n1n2 - d2) / (n1n2_2 - 1);

  auto point = n1 * a + n2 * b;

  l.setPoint(point);
  direction.normalize();
  l.setDirection(direction);

  return true;
}

} // namespace jmk
