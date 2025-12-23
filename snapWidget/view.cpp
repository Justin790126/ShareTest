#include "view.h"
#include <QtGui/QMouseEvent>
#include <QtGui/QPainter>

static Shape makePolylineDbu(const std::vector<Point_C> &pts,
                             bool closed = false) {
  Shape s;
  s.closed = closed;
  s.pa.pts = pts;
  return s;
}

DemoView::DemoView(QWidget *parent)
    : QWidget(parent), umPerPixel(0.01) // 1 px = 0.01 um (demo)
      ,
      originPx(20, 20), m_layout(), m_selector(&m_layout), m_usrUm(0, 0),
      m_snapUm(0, 0), m_res(SR_None) {
  setMouseTracking(true);

  // demo: set dbu scale (1 dbu = 0.001 um => 1nm)
  m_selector.umPerDBU = 0.001;
  m_selector.edgeSnapR_um = 0.03; // 30nm demo

  // Build fake layout in DBU:
  // Rectangle polygon (closed)
  // 0.5um ~ 4um region
  auto U = [&](double um) { return roundToLong(um / m_selector.umPerDBU); };

  m_layout.shapes.push_back(makePolylineDbu(
      {
          Point_C(U(0.5), U(0.5)),
          Point_C(U(4.0), U(0.5)),
          Point_C(U(4.0), U(2.5)),
          Point_C(U(0.5), U(2.5)),
      },
      true));

  // A polyline "L" shape
  m_layout.shapes.push_back(
      makePolylineDbu({Point_C(U(1.0), U(3.2)), Point_C(U(3.5), U(3.2)),
                       Point_C(U(3.5), U(4.5))},
                      false));

  // Another diagonal segment
  m_layout.shapes.push_back(makePolylineDbu(
      {Point_C(U(0.8), U(4.8)), Point_C(U(4.2), U(5.6))}, false));

  // Very short edges (few nm)
  m_layout.shapes.push_back(makePolylineDbu(
      {Point_C(U(6.0), U(5.5)), Point_C(U(6.002), U(5.5)), // 2nm edge
       Point_C(U(6.5), U(5.5))},
      false));

  // Duplicate points / zero-length edges
  m_layout.shapes.push_back(makePolylineDbu(
      {Point_C(U(9.0), U(1.0)), Point_C(U(9.0), U(1.0)), // duplicate
       Point_C(U(10.0), U(1.0)), Point_C(U(10.0), U(2.0)),
       Point_C(U(10.0), U(2.0)), // duplicate
       Point_C(U(9.0), U(2.0))},
      true));

  // Concave polygon with notch
  m_layout.shapes.push_back(makePolylineDbu(
      {Point_C(U(1.0), U(6.0)), Point_C(U(4.5), U(6.0)),
       Point_C(U(4.5), U(5.0)), Point_C(U(2.5), U(5.0)), // notch inward
       Point_C(U(2.5), U(4.0)), Point_C(U(4.5), U(4.0)),
       Point_C(U(4.5), U(3.0)), Point_C(U(1.0), U(3.0))},
      true)); // closed polygon
  // Z-shaped polyline (routing-like)
  m_layout.shapes.push_back(makePolylineDbu(
      {
          Point_C(U(5.0), U(1.0)), Point_C(U(7.0), U(1.0)), // horizontal
          Point_C(U(7.0), U(2.0)),                          // vertical
          Point_C(U(6.0), U(2.0)),                          // horizontal back
          Point_C(U(6.0), U(3.5)),                          // vertical
          Point_C(U(8.0), U(3.5))                           // horizontal
      },
      false));
  // Rectangle intersecting the concave polygon (through the notch)
  m_layout.shapes.push_back(makePolylineDbu(
      {
          Point_C(U(2.0), U(4.3)), // left-bottom
          Point_C(U(5.2), U(4.3)), // right-bottom (extends outside polygon)
          Point_C(U(5.2), U(4.7)), // right-top
          Point_C(U(2.0), U(4.7))  // left-top
      },
      true)); // closed rectangle

  // ===== (Hexagon rotated 30deg) + (center rectangle intersects) =====
  double cx = 7.0; // center x (um)
  double cy = 7.0; // center y (um)
  double R = 1.5;  // hex radius (um), center-to-vertex

  const double s = 0.8660254; // sqrt(3)/2

  // Rotated 30° regular hexagon (pointy-top)
  // vertices at: (0,90,150,210,270,330) degrees in a flat-top frame,
  // or equivalently this explicit form:
  m_layout.shapes.push_back(
      makePolylineDbu({Point_C(U(cx), U(cy + R)), // top
                       Point_C(U(cx + R * s), U(cy + R * 0.5)),
                       Point_C(U(cx + R * s), U(cy - R * 0.5)),
                       Point_C(U(cx), U(cy - R)), // bottom
                       Point_C(U(cx - R * s), U(cy - R * 0.5)),
                       Point_C(U(cx - R * s), U(cy + R * 0.5))},
                      true)); // closed

  // Center rectangle that intersects the hexagon
  // Make it wide enough to cut through left/right slanted edges
  double rectHalfW = 1.8;  // (um) half width  -> total width 3.6um
  double rectHalfH = 0.35; // (um) half height -> total height 0.7um

  m_layout.shapes.push_back(
      makePolylineDbu({Point_C(U(cx - rectHalfW), U(cy - rectHalfH)),
                       Point_C(U(cx + rectHalfW), U(cy - rectHalfH)),
                       Point_C(U(cx + rectHalfW), U(cy + rectHalfH)),
                       Point_C(U(cx - rectHalfW), U(cy + rectHalfH))},
                      true)); // closed rectangle
}

QPointF DemoView::screenToUm(const QPointF &p) const {
  return QPointF((p.x() - originPx.x()) * umPerPixel,
                 (p.y() - originPx.y()) * umPerPixel);
}

QPointF DemoView::umToScreen(const QPointF &p) const {
  return QPointF(originPx.x() + p.x() / umPerPixel,
                 originPx.y() + p.y() / umPerPixel);
}

void DemoView::mouseMoveEvent(QMouseEvent *e) {
  m_usrUm = screenToUm(e->pos());
  m_selector.GetClosestEdgeVertex(m_usrUm, m_snapUm, m_edge, m_vertex, m_res);
  update();
}

void DemoView::paintEvent(QPaintEvent *) {
  QPainter qp(this);
  qp.fillRect(rect(), Qt::white);
  qp.setRenderHint(QPainter::Antialiasing, true);

  // Draw shapes
  qp.setPen(QPen(Qt::black, 1));
  for (size_t si = 0; si < m_layout.shapes.size(); ++si) {
    const Shape &s = m_layout.shapes[si];
    const int n = s.pa.GetPointCount();
    if (n < 2)
      continue;

    QPainterPath path;
    for (int i = 0; i < n; ++i) {
      const Point_C p = s.pa[i];
      QPointF pum((double)p.x() * m_selector.umPerDBU,
                  (double)p.y() * m_selector.umPerDBU);
      QPointF ps = umToScreen(pum);
      if (i == 0)
        path.moveTo(ps);
      else
        path.lineTo(ps);
    }
    if (s.closed)
      path.closeSubpath();
    qp.drawPath(path);
  }

  // Draw user point
  QPointF usrS = umToScreen(m_usrUm);
  qp.setPen(QPen(Qt::blue, 2));
  qp.drawEllipse(usrS, 4, 4);

  // Draw snap point
  QPointF snapS = umToScreen(m_snapUm);
  qp.setPen(QPen(Qt::red, 2));
  qp.drawEllipse(snapS, 4, 4);

  // Draw highlight based on res
  if (m_res == SR_Corner) {
    qp.setPen(QPen(Qt::red, 2));
    qp.drawEllipse(umToScreen(m_vertex.posUm), 7, 7);
  } else if (m_res == SR_Edge) {
    qp.setPen(QPen(Qt::darkRed, 2));
    qp.drawLine(umToScreen(m_edge.p0Um), umToScreen(m_edge.p1Um));
    qp.setPen(QPen(Qt::darkGreen, 2));
    qp.drawEllipse(umToScreen(m_edge.projUm), 5, 5);
  }

  // HUD
  qp.setPen(Qt::black);
  QString resStr =
      (m_res == SR_Corner) ? "Corner" : (m_res == SR_Edge) ? "Edge" : "None";
  qp.drawText(10, height() - 40,
              QString("usr(um)=(%1,%2)  snap=(%3,%4)  res=%5")
                  .arg(m_usrUm.x(), 0, 'f', 3)
                  .arg(m_usrUm.y(), 0, 'f', 3)
                  .arg(m_snapUm.x(), 0, 'f', 3)
                  .arg(m_snapUm.y(), 0, 'f', 3)
                  .arg(resStr));
}
