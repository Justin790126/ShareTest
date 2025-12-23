#pragma once
#include <vector>

// ---- DBU geometry ----
struct Point_C {
    long _x, _y;
    Point_C() : _x(0), _y(0) {}
    Point_C(long x, long y) : _x(x), _y(y) {}
    long x() const { return _x; }
    long y() const { return _y; }
};

struct PointArray_C {
    std::vector<Point_C> pts;
    int GetPointCount() const { return (int)pts.size(); }
    const Point_C& operator[](int i) const { return pts[i]; }
};

struct Shape {
    PointArray_C pa;
    bool closed; // polygon?
    Shape() : closed(false) {}
};

// ---- Hit operation base ----
struct HitOperation {
    virtual ~HitOperation() {}
    virtual void Act(const Shape& shp) = 0;
};

// ---- "Layout" + traverse simulation ----
struct FakeLayout {
    std::vector<Shape> shapes;

    template <typename RegionFilterFn>
    void Traverse(const RegionFilterFn& regionFilter, HitOperation& hop) const {
        for (size_t i = 0; i < shapes.size(); ++i) {
            const Shape& s = shapes[i];
            if (regionFilter(s)) {
                hop.Act(s);
            }
        }
    }
};
