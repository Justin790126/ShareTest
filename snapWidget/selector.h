#pragma once
#include <QtCore/QPointF>
#include "layout.h"

enum SearchRes {
    SR_None   = 0,
    SR_Edge   = 1,
    SR_Corner = 2
};

struct Vertex {
    QPointF posUm;
    Vertex() {}
    explicit Vertex(const QPointF& p) : posUm(p) {}
};

struct Edge {
    QPointF p0Um;
    QPointF p1Um;
    QPointF projUm;
    Edge() {}
    Edge(const QPointF& a, const QPointF& b, const QPointF& q) : p0Um(a), p1Um(b), projUm(q) {}
};

struct SnapState {
    // input (um)
    double umPerDBU;
    double px_um;
    double py_um;

    // params (um)
    double cornerR_um; // 0.004
    double edgeR_um;   // configurable

    // best corner
    bool   hasCorner;
    double bestCornerD2;
    QPointF bestCornerUm;

    // best edge
    bool   hasEdge;
    double bestEdgeD2;
    QPointF bestEdgeProjUm;
    QPointF bestEdgeP0Um;
    QPointF bestEdgeP1Um;

    SnapState()
      : umPerDBU(0.0), px_um(0.0), py_um(0.0),
        cornerR_um(0.004), edgeR_um(0.02),
        hasCorner(false), bestCornerD2(1e300),
        hasEdge(false), bestEdgeD2(1e300) {}
};

class lcSelector;

class SelectorHitOp : public HitOperation {
public:
    SelectorHitOp(lcSelector* sel, SnapState* st) : m_sel(sel), m_st(st) {}
    virtual void Act(const Shape& shp);

private:
    lcSelector* m_sel;
    SnapState*  m_st;
};

class lcSelector {
public:
    explicit lcSelector(const FakeLayout* layout);

    // config
    double umPerDBU;      // e.g. 1 dbu = 0.001 um
    double edgeSnapR_um;  // e.g. 0.02 um (20nm)

    // called by mouse move
    void GetClosestEdgeVertex(const QPointF& usrPtum,
                              QPointF& snapPtum,
                              Edge& edge,
                              Vertex& vertex,
                              int& searchRes);

    // called by Act() for each hit shape
    void CheckShape_Update(const Shape& shp, SnapState& st);

private:
    const FakeLayout* m_layout;
};
