#ifndef POLYGON_H
#define POLYGON_H

#include "Point.h"
#include "Vector.h"
#include "PolygonDCEL.h"

#include <list>
#include <vector>

namespace jmk {

template <class T, size_t dim> 
struct Vertex {
  jmk::Vector<T, dim> point;
  Vertex *next;
  Vertex *prev;
  bool is_ear = false;
  bool is_processed = false;

  Vertex() {}

  Vertex(jmk::Vector<T, dim> &_point, Vertex<T, dim> *_next = nullptr,
         Vertex<T, dim> *_prev = nullptr)
      : point(_point), next(_next), prev(_prev) {}
};

typedef Vertex<float, DIM2> Vertex2d;


template <class T, size_t dim = DIM3> 
class Polygon {
  std::vector<Vertex<T, dim> *> vertex_list;

public:
  Polygon(std::list<jmk::Vector<T, dim>> &points) {
    const int size = points.size();
    if (size < 3) {
      std::cout << "Not enough points to construct a polygon\n";
      return;
    }

    // Step 1: Create vertices without setting next/prev
    for (auto point : points) {
      vertex_list.push_back(new Vertex<T, dim>(point));
    }

    // Step 2: Set next and prev pointers
    for (size_t i = 0; i < size; i++) {
      vertex_list[i]->next = vertex_list[(i + 1) % size];
      vertex_list[i]->prev = vertex_list[(i - 1 + size) % size];
    }
  }

  // Destructor to clean up dynamically allocated vertices
  ~Polygon() {
    for (auto vertex : vertex_list) {
      delete vertex;
    }
  }

  std::vector<Vertex<T, dim> *> getVertices() { return vertex_list; }
};

typedef Polygon<float, DIM3> PolygonS3d;
typedef Polygon<float, DIM2> PolygonS2d;

template<class T, size_t dim>
struct Edge {
  Vertex<T, dim> v1;
  Vertex<T, dim> v2;

  Edge(Vertex<T, dim> _v1, Vertex<T, dim> _v2) : v1(_v1), v2(_v2) {

  }

};

typedef Edge<float, DIM2> Edge2d;

} // namespace jmk

#endif /* POLYGON_H */