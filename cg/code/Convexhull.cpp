#include "Angle.h"
#include "Point.h"
#include "Convexhull.h"
#include "GeoUtils.h"
#include <list>
#include <vector>
#include <cmath>
#include <algorithm>


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

void convexhull2DModifiedGrahams(vector<Point2d>& _points, vector<Point2d>& _convex)
{
  if (_points.size() <= 3) {
      return;
  }
  // is convex push in stack
  // is  reflect pop stack, check angle

  // sort by x coordinate
  std::sort(_points.begin(), _points.end(), [](const Point2d& a, const Point2d& b) {
      return a[X] < b[X] || (a[X] == b[X] && a[Y] < b[Y]);
  });

  std::vector<Point2d> l_upper;
  std::vector<Point2d> l_lower;

  l_upper.push_back(*_points.begin());
  l_upper.push_back(*(std::next(_points.begin())));

  int index = 0;
  for (size_t i = 2; i < _points.size(); i++) {
    index = l_upper.size();
    const auto& next_point = _points[i];
    while (l_upper.size() > 1 && left(l_upper[index-2], l_upper[index-1], next_point)) {
      /*
        A     C (next_point)  > 180 degree ---> pop back
         |   /
         |  /
         | /
         B *
      
      */
      l_upper.pop_back();
      index = l_upper.size();
    }
    l_upper.push_back(next_point);
  }

  std::reverse(_points.begin(), _points.end());

  l_lower.push_back(*_points.begin());
  l_lower.push_back(*(std::next(_points.begin())));

  index = 0;
  for (size_t i = 2; i < _points.size(); i++) {
    index = l_lower.size();
    const auto& next_point = _points[i];
    while (l_lower.size() > 1 && left(l_lower[index-2], l_lower[index-1], next_point)) {
      l_lower.pop_back();
      index = l_lower.size();
    }
    l_lower.push_back(next_point);
  }
  l_lower.pop_back();

  _convex.insert(_convex.end(), l_upper.begin(), l_upper.end());
  _convex.insert(_convex.end(), l_lower.begin(), l_lower.end());
}

static void adjust_normal(Face* _face, Point3d& _ref_point)
{
  int order = FaceVisibility(*_face, _ref_point);
  if (order < 0) {
    _face->normal_switch_needed = true;
  }
}

static bool incident_face(Face* _face, Edge3d* _edge) {
    // Get the vertices of the edge
    Vertex3d* v1 = _edge->vertices[0];
    Vertex3d* v2 = _edge->vertices[1];

    // Get the vertices of the face
    Vertex3d* fv1 = _face->vertices[0];
    Vertex3d* fv2 = _face->vertices[1];
    Vertex3d* fv3 = _face->vertices[2];

    // Check if both vertices of the edge are among the face's vertices
    bool v1_found = (v1 == fv1 || v1 == fv2 || v1 == fv3);
    bool v2_found = (v2 == fv1 || v2 == fv2 || v2 == fv3);

    // The edge is incident to the face if both of its vertices are in the face
    return v1_found && v2_found;
}

static void construct_initial_polyhedron(vector<Vertex3d*>& _points, int i, vector<Face*>& faces, 
  vector<Edge3d*>& edges, Point3d& ref_point)
{
  faces.push_back(new Face(_points[i+0], _points[i+1], _points[i+2]));
  faces.push_back(new Face(_points[i+0], _points[i+1], _points[i+3]));
  faces.push_back(new Face(_points[i+1], _points[i+2], _points[i+3]));
  faces.push_back(new Face(_points[i+2], _points[i+0], _points[i+3]));

  for (size_t i = 0; i < faces.size(); i++) {
    adjust_normal(faces[i], ref_point);
  }

  edges.push_back(new Edge3d(_points[i+0], _points[i+1]));
  edges.push_back(new Edge3d(_points[i+1], _points[i+2]));
  edges.push_back(new Edge3d(_points[i+2], _points[i+0]));
  edges.push_back(new Edge3d(_points[i+0], _points[i+3]));
  edges.push_back(new Edge3d(_points[i+1], _points[i+3]));
  edges.push_back(new Edge3d(_points[i+2], _points[i+3]));

  faces[0]->addEdge(edges[0]);
  faces[0]->addEdge(edges[1]);
  faces[0]->addEdge(edges[2]);

  faces[1]->addEdge(edges[0]);
  faces[1]->addEdge(edges[3]);
  faces[1]->addEdge(edges[4]);

  faces[2]->addEdge(edges[1]);
  faces[2]->addEdge(edges[4]);
  faces[2]->addEdge(edges[5]);

  faces[3]->addEdge(edges[2]);
  faces[3]->addEdge(edges[5]);
  faces[3]->addEdge(edges[3]);

  edges[0]->faces[0] = faces[0];
  edges[0]->faces[1] = faces[1];

  edges[1]->faces[0] = faces[0];
  edges[1]->faces[1] = faces[2];

  edges[2]->faces[0] = faces[0];
  edges[2]->faces[1] = faces[3];

  edges[3]->faces[0] = faces[1];
  edges[3]->faces[1] = faces[3];

  edges[4]->faces[0] = faces[1];
  edges[4]->faces[1] = faces[2];

  edges[5]->faces[0] = faces[3];
  edges[5]->faces[1] = faces[2];
}

void convexhull3D(vector<Point3d>& _points, vector<Face*>& faces)
{
  vector<Vertex3d*> vertices;
  for (size_t i = 0; i < _points.size(); i++) {
    vertices.push_back(new Vertex3d(&_points[i]));
  } 

  vector<Edge3d*> edges;

  size_t i = 0, j = 0;
  bool found_noncoplaner = false;
  for (i = 0; i < _points.size() -3 ; i++) {
    if (!coplaner(_points[i], _points[i+1], _points[i+2], _points[i+3])) {
      found_noncoplaner = true;
      break;
    }
  }

  if (!found_noncoplaner) {
    std::cout << "All points are coplaner" << std::endl;
    return;
  }

  float x_mean = (_points[i][X] + _points[i+1][X] + _points[i+2][X] + _points[i+3][X]) / 4.0f;
  float y_mean = (_points[i][Y] + _points[i+1][Y] + _points[i+2][Y] + _points[i+3][Y]) / 4.0f;
  float z_mean = (_points[i][Z] + _points[i+1][Z] + _points[i+2][Z] + _points[i+3][Z]) / 4.0f;
  float x_p = x_mean;
  float y_p = y_mean;
  float z_p = z_mean;
  Point3d point_ref(x_p, y_p, z_p);
  construct_initial_polyhedron(vertices, i, faces, edges, point_ref);

  vertices[i]->processed = true;
  vertices[i+1]->processed = true;
  vertices[i+2]->processed = true;
  vertices[i+3]->processed = true;

  for (size_t i = 0; i < vertices.size(); i++)
  {
    if (vertices[i]->processed) continue;
  
    vector<Face*> visible_faces;
    vector<Edge3d*> border_edges;
    vector<Edge3d*> tobe_deleted_edges;

    // point has not yet processed and it is outside the current hull
    for (size_t j = 0; j < faces.size(); j++) {

      float volum = FaceVisibility(*faces[j], *(vertices[i]->point));

      if ((!faces[j]->normal_switch_needed && volum<0)
        || (faces[j]->normal_switch_needed && volum>0)) {
          faces[j]->visible = true;
          visible_faces.push_back(faces[j]);
        }

    }

    if (!visible_faces.size()) continue;

    for (size_t k = 0; k < visible_faces.size(); k++) {
      for (size_t j = 0; j < visible_faces[k]->edges.size(); j++) {
        if (visible_faces[k]->edges[j]->faces[0]->visible &&
            visible_faces[k]->edges[j]->faces[1]->visible) {
          tobe_deleted_edges.push_back(visible_faces[k]->edges[j]);
        } else {
          border_edges.push_back(visible_faces[k]->edges[j]);
        }
      }
    }

    vector<Face*> new_faces;
    vector<Edge3d*> new_edges;

    const unsigned int new_size = border_edges.size();

    // find unique points in border edges
    list<Vertex3d*> unque_vertices;
    for (size_t j = 0; j < new_size; j++) {
      unque_vertices.push_back(border_edges[j]->vertices[0]);
      unque_vertices.push_back(border_edges[j]->vertices[1]);
    }

    unque_vertices.sort();
    unque_vertices.unique([](Vertex3d* a, Vertex3d* b) { return *(a->point) == *(b->point); });
    std::list<Vertex3d*>::iterator it;
    for (size_t j = 0; j < new_size; j++) {
      it = unque_vertices.begin();
      std::advance(it, j);

      new_edges.push_back(new Edge3d(*it, vertices[i]));
      new_faces.push_back(new Face(border_edges[j]->vertices[0], border_edges[j]->vertices[1], vertices[i]));

      if (border_edges[j]->faces[0]->visible) {
        border_edges[j]->faces[0] = new_faces[new_faces.size()-1];
      } else if (border_edges[j]->faces[1]->visible) {
        border_edges[j]->faces[1] = new_faces[new_faces.size()-1];
      }
    }

    for (size_t j = 0; j < new_size; j++) {
      adjust_normal(new_faces[j], point_ref);
    }

    for (size_t j = 0; j < new_size; j++) {
      vector<Face*> incident_faces;
      for (size_t k = 0; k < new_faces.size(); k++) {
        if (incident_face(new_faces[k], new_edges[j])) {
          incident_faces.push_back(new_faces[k]);
        }

        new_edges[j]->faces[0] = incident_faces[0];
        new_edges[j]->faces[1] = incident_faces[1];
      }

    }

    for (size_t k = 0; k < tobe_deleted_edges.size(); k++) {
      for (size_t j = 0; j < edges.size(); j++) {
        if (*tobe_deleted_edges[k] == *edges[j]) {
          edges.erase(edges.begin() + j);
        }
      }
    }

    for (size_t k = 0; k < visible_faces.size(); k++) {
      for (size_t j = 0; j < faces.size() ; j++) {
        if (visible_faces[k] == faces[j]) {
          faces.erase(faces.begin() + j);
        }
      }
    }

    faces.insert(faces.end(), new_faces.begin(), new_faces.end());
    edges.insert(edges.end(), new_edges.begin(), new_edges.end());
  }
}

} // namespace jmk