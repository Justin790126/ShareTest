#include "selector.h"
#include <cmath>

static inline double dot2(double ax, double ay, double bx, double by) {
    return ax*bx + ay*by;
}
static inline double clamp01(double t) {
    if (t < 0.0) return 0.0;
    if (t > 1.0) return 1.0;
    return t;
}

lcSelector::lcSelector(const FakeLayout* layout)
  : umPerDBU(0.001)        // demo: 1 dbu = 0.001 um (1nm)
  , edgeSnapR_um(0.02)     // demo: 20nm
  , m_layout(layout)
{}

void SelectorHitOp::Act(const Shape& shp) {
    m_sel->CheckShape_Update(shp, *m_st);
}

void lcSelector::CheckShape_Update(const Shape& shp, SnapState& st)
{
    const PointArray_C& array = shp.pa;
    const int n = array.GetPointCount();
    if (n < 2) return;

    const double cornerR2 = st.cornerR_um * st.cornerR_um;
    const double edgeR2   = st.edgeR_um   * st.edgeR_um;

    // -------------------------
    // 1) Corner pass (priority)
    // -------------------------
    for (int i = 0; i < n; ++i) {
        const Point_C p = array[i];

        const double vx = (double)p.x() * st.umPerDBU;
        const double vy = (double)p.y() * st.umPerDBU;

        const double dx = st.px_um - vx;
        const double dy = st.py_um - vy;
        const double d2 = dx*dx + dy*dy;

        if (d2 <= cornerR2 && d2 < st.bestCornerD2) {
            st.hasCorner = true;
            st.bestCornerD2 = d2;
            st.bestCornerUm = QPointF(vx, vy);
        }
    }

    // 如果已經命中 corner，就不再更新 edge（corner 優先）
    if (st.hasCorner) return;

    // -------------------------
    // 2) Edge pass (fallback)
    // -------------------------
    const int edgeCount = shp.closed ? n : (n - 1);
    for (int i = 0; i < edgeCount; ++i) {
        const int j = (shp.closed) ? ((i + 1) % n) : (i + 1);

        const Point_C pt0 = array[i];
        const Point_C pt1 = array[j];

        const double x0 = (double)pt0.x() * st.umPerDBU;
        const double y0 = (double)pt0.y() * st.umPerDBU;
        const double x1 = (double)pt1.x() * st.umPerDBU;
        const double y1 = (double)pt1.y() * st.umPerDBU;

        const double abx = x1 - x0;
        const double aby = y1 - y0;
        const double ab2 = abx*abx + aby*aby;
        if (ab2 == 0.0) continue;

        const double apx = st.px_um - x0;
        const double apy = st.py_um - y0;

        double t = dot2(apx, apy, abx, aby) / ab2;
        t = clamp01(t); // 允許用 clamp

        const double qx = x0 + t * abx;
        const double qy = y0 + t * aby;

        const double dx = st.px_um - qx;
        const double dy = st.py_um - qy;
        const double d2 = dx*dx + dy*dy;

        if (d2 <= edgeR2 && d2 < st.bestEdgeD2) {
            st.hasEdge = true;
            st.bestEdgeD2 = d2;
            st.bestEdgeProjUm = QPointF(qx, qy);
            st.bestEdgeP0Um   = QPointF(x0, y0);
            st.bestEdgeP1Um   = QPointF(x1, y1);
        }
    }
}

void lcSelector::GetClosestEdgeVertex(const QPointF& usrPtum,
                                      QPointF& snapPtum,
                                      Edge& edge,
                                      Vertex& vertex,
                                      int& searchRes)
{
    SnapState st;
    st.umPerDBU    = this->umPerDBU;
    st.px_um       = usrPtum.x();
    st.py_um       = usrPtum.y();
    st.cornerR_um  = 0.004;          // 4nm
    st.edgeR_um    = this->edgeSnapR_um;

    SelectorHitOp hop(this, &st);

    // region filter：demo 直接全收；你可以改成 bbox/ROI
    struct AllPass {
        bool operator()(const Shape&) const { return true; }
    } filter;

    m_layout->Traverse(filter, hop);

    // finalize：Corner > Edge > None
    if (st.hasCorner) {
        snapPtum = st.bestCornerUm;
        vertex = Vertex(st.bestCornerUm);
        searchRes = SR_Corner;
        return;
    }
    if (st.hasEdge) {
        snapPtum = st.bestEdgeProjUm;
        edge = Edge(st.bestEdgeP0Um, st.bestEdgeP1Um, st.bestEdgeProjUm);
        searchRes = SR_Edge;
        return;
    }

    // None => 回原點
    snapPtum = usrPtum;
    searchRes = SR_None;
}
