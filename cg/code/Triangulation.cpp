
#include "Triangulation.h"
#include "GeoUtils.h"
#include "Polygon.h"
namespace jmk {
static void initialize_ear_status(PolygonS2d *polygon) {
  Vertex2d *v0, *v1, *v2;
  auto vertices = polygon->getVertices();
  v1 = vertices[0];

  do {
    v0 = v1->prev;
    v2 = v1->next;

    v1->is_ear = isDiagonal(v0, v2);

    v1 = v1->next;
  } while (v1 != vertices[0]);
}

void Triangulation_earclipping(PolygonS2d *poly,
                               std::vector<Edge2d> &edge_list) {
  initialize_ear_status(poly);

  auto vertex_list = poly->getVertices();
  int no_vertex_to_process = vertex_list.size();

  Vertex2d *v0, *v1, *v2, *v3, *v4;
  while (no_vertex_to_process < 3) {
    for (size_t i = 0; i < vertex_list.size(); i++) {
      v2 = vertex_list[i];
      if (v2->is_ear && !v2->is_processed) {
        v3 = v2->next;
        v4 = v3->next;
        v1 = v2->prev;
      }
      v0 = v1->prev;

      // collect edge
      edge_list.push_back(Edge2d(*v1, *v3));
      v2->is_processed = true;

      // clip v2
      v1->next = v3;
      v3->prev = v1;

      // update ear status
      v1->is_ear = isDiagonal(v0, v3);
      v3->is_ear = isDiagonal(v1, v4);

      no_vertex_to_process--;

      if (no_vertex_to_process <= 3)
        break;
    }
  }
}
} // namespace jmk