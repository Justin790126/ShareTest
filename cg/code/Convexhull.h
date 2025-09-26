#ifndef CONVEX_HULL_H
#define CONVEX_HULL_H

#include "Polygon.h"
#include "Point.h"
#include "Plane.h"

namespace jmk {

    struct Face;

    struct Vertex3d {
        Point3d* point = nullptr;
        bool processed = false;

        Vertex3d() {}

        Vertex3d(Point3d* _point) {
            point = _point;
            processed = false;
        }

    };

    struct Edge3d {
        Vertex3d* vertices[2];
        Face* faces[2] = { nullptr, nullptr };
        Edge3d(Vertex3d* p1, Vertex3d* p2) {
            vertices[0] = p1;
            vertices[1] = p2;
        }
        bool operator==(const Edge3d& _other) {
            return (vertices[0] == _other.vertices[0] && vertices[1] == _other.vertices[1]) ||
                (vertices[0] == _other.vertices[1] && vertices[1] == _other.vertices[0]);
        }
    };

    struct Face {
        std::vector<Edge3d*> edges;
        std::vector<Vertex3d*> vertices;
        Planef plane;
        bool visible = false;
        bool normal_switch_needed = false;

        Face() {}
        Face(Vertex3d* p1, Vertex3d* p2, Vertex3d* p3) {
            vertices.push_back(p1);
            vertices.push_back(p2);
            vertices.push_back(p3);
            edges.push_back(new Edge3d(p1, p2));
            edges.push_back(new Edge3d(p2, p3));
            edges.push_back(new Edge3d(p3, p1));
            plane = Planef(*p1->point, *p2->point, *p3->point);
            visible = false;
            normal_switch_needed = false;
        }
        bool operator==(const Face& _other) {
            return (vertices[0] == _other.vertices[0] && vertices[1] == _other.vertices[1] && vertices[2] == _other.vertices[2]) ||
                (vertices[0] == _other.vertices[1] && vertices[1] == _other.vertices[2] && vertices[2] == _other.vertices[0]) ||
                (vertices[0] == _other.vertices[2] && vertices[1] == _other.vertices[0] && vertices[2] == _other.vertices[1]);
        }
        void addEdge(Edge3d* edg_ptr) {
            edges.push_back(edg_ptr);
        }
    };

};

#endif /* CONVEX_HULL_H */