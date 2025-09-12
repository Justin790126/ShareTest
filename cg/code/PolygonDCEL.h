#ifndef POLYGON_DCEL_H
#define POLYGON_DCEL_H

#include "Vector.h"
#include <iostream>
#include <vector>

using namespace std;

namespace jmk {

    static int _id = 1;

    template<class type, size_t dim>
    struct EdgeDCEL;

    template<class type, size_t dim>
    struct FaceDCEL;

    template<class type=float, size_t dim = DIM3>
    struct VertexDCEL {
        Vector<float, dim> point;
        EdgeDCEL<type, dim>* incident_edge = NULL;

        VertexDCEL(Vector<type, dim>& _point) : point(_point) {}

        void print() {
            cout << "(" << point[X] << ", " << point[Y] << "\n";
        }
    };

    template<class type=float, size_t dim = DIM3>
    struct EdgeDCEL {
        VertexDCEL<type, dim>* origin = NULL;
        EdgeDCEL<type, dim>* twin = NULL;
        EdgeDCEL<type, dim>* next = NULL;
        EdgeDCEL<type, dim>* prev = NULL;
        FaceDCEL<type, dim>* incident_face = NULL;
        int id;

        EdgeDCEL() { id = -1; }

        EdgeDCEL(VertexDCEL<type, dim>* _origin) : origin(_origin) {
            id = _id++;
        }

        VertexDCEL<type, dim>* destination() {
            return twin->origin;
        }

        void print() 
        {
            cout << "This point pointer" << this << "\n";
            cout << "Origin: "; this->origin->print();
            cout << "Twin pointer" << this->twin << "\n";
            cout << "Next pointer" << this->next << "\n";
            cout << "Prev pointer" << this->prev << "\n";
        }

    };

    template<class type=float, size_t dim = DIM3>
    struct FaceDCEL {
        EdgeDCEL<type, dim>* outer = NULL;
        vector<EdgeDCEL<type, dim>*> inner;
        FaceDCEL() {}

        void print() {

        }

        vector<EdgeDCEL<type, dim>*> getEdgeList() {
            vector<EdgeDCEL<type, dim>*> edge_list;
            if (outer == NULL) return edge_list;
            
            auto outer_edge_ptr = outer;
            auto outer_next_ptr = outer->next;

            edge_list.push_back(outer_edge_ptr);
            while (outer_next_ptr != outer_edge_ptr) {
                edge_list.push_back(outer_next_ptr);
                outer_next_ptr = outer_next_ptr->next;
            }
            return edge_list;
        }

        vector<Vector<float, dim>> getPoints() {
            vector<Vector<float, dim>> points;
            if (outer == NULL) return points;

            auto edge_ptr = outer;
            points.push_back(edge_ptr->origin->point);

            auto next_ptr = edge_ptr->next;
            while (next_ptr != outer) {
                points.push_back(next_ptr->origin->point);
                next_ptr = next_ptr->next;
            }
            return points;
        }
    };

    template<class type=float, size_t dim = DIM3>
    class PolygonDCEL {
        typedef Vector<type, dim> VectorNf;
        vector<VertexDCEL<type, dim>*> vertex_list;
        vector<EdgeDCEL<type, dim>*> edge_list;
        vector<FaceDCEL<type, dim>*> face_list;

        EdgeDCEL<type, dim>* empty_edge = new EdgeDCEL<type, dim>();

    public:

        explicit PolygonDCEL(vector<VectorNf>&);

    };

    typedef VertexDCEL<float, 2U> Vertex2dDCEL;
    typedef EdgeDCEL<float, 2U> Edge2dDCEL;
    typedef PolygonDCEL<float, 2U> Polygon2dDCEL;

    template<class type, size_t dim>
    inline PolygonDCEL<type, dim>::PolygonDCEL(std::vector<VectorNf>& _points)
    {
        int size = _points.size();
        if (size < 3) {
            return;
        }

        for (size_t i = 0; i < _points.size(); i++) {
            vertex_list.push_back(new VertexDCEL<type, dim>(_points[i]));
        }

        for (size_t i = 0; i <= vertex_list.size() - 2; i++) {
            auto hfedge = new EdgeDCEL<type, dim>(vertex_list[i]);
            auto edge_twin = new EdgeDCEL<type, dim>(vertex_list[i + 1]);
            vertex_list[i]->incident_edge = hfedge;
            hfedge->twin = edge_twin;
            edge_twin->twin = hfedge;
            edge_list.push_back(hfedge);
            edge_list.push_back(edge_twin);
        }

        auto hfedge = new EdgeDCEL<type, dim>(vertex_list.back());
        auto edge_twin = new EdgeDCEL<type, dim>(vertex_list.front());
        hfedge->twin = edge_twin;
        edge_twin->twin = hfedge;
        edge_list.push_back(hfedge);

        vertex_list[vertex_list.size() -1]->incident_edge = hfedge;

        for (size_t i = 2; i <= edge_list.size()-3; i++) {
            if (i%2 == 0) {
                edge_list[i]->next = edge_list[i + 2];
                edge_list[i]->prev = edge_list[i - 2];
            }
            else {
                edge_list[i]->next = edge_list[i - 2];
                edge_list[i]->prev = edge_list[i + 2];
            }
        }

        edge_list[0]->next = edge_list[2];
        edge_list[0]->prev = edge_list[edge_list.size() - 2];
        edge_list[1]->next = edge_list[edge_list.size() - 1];
        edge_list[1]->prev = edge_list[3];

        edge_list[edge_list.size() -2]->next = edge_list[0];
        edge_list[edge_list.size() -2]->prev = edge_list[edge_list.size() - 4];
        edge_list[edge_list.size() -1]->next = edge_list[edge_list.size() - 3];
        edge_list[edge_list.size() -1]->prev = edge_list[1];

        FaceDCEL<type, dim>* f1 = new FaceDCEL<type, dim>();
        FaceDCEL<type, dim>* f2 = new FaceDCEL<type, dim>();

        f1->outer = edge_list[0]; // counter clockwise
        f2->inner.push_back(edge_list[1]); // clockwise

        face_list.push_back(f1);
        face_list.push_back(f2);

        f1->outer->incident_face = f1;
        EdgeDCEL<type, dim>* edge = f1->outer->next;
        while (edge != f1->outer) {
            edge->incident_face = f1;
            edge = edge->next;
        }

        f2->inner[0]->incident_face = f2;
        edge = f2->inner[0]->next;
        while (edge != f2->inner[0]) {
            edge->incident_face = f2;
            edge = edge->next;
        }

    }



};


#endif /* POLYGON_DCEL_H */