#include "LayoutCanvas.h"

#include <QMouseEvent>
#include <QPainter>
#include <QtGlobal>

LayoutCanvas::LayoutCanvas(QWidget* parent)
    : QWidget(parent),
      m_iMode(Mode_Select),
      m_pressed(false),
      m_dragging(false),
      m_dragThreshold(3),
      m_pressButton(Qt::NoButton),
      m_lastPreviewRect()
{
    setMouseTracking(true);
    setFocusPolicy(Qt::StrongFocus);
}

void LayoutCanvas::setMode(int m)
{
    if (m_iMode == m) return;
    m_iMode = m;
    emit modeChanged(m_iMode);

    // cancel ongoing gesture on mode change
    m_pressed = false;
    m_dragging = false;
    m_pressButton = Qt::NoButton;
    m_lastPreviewRect = QRect();

    update();
}

int LayoutCanvas::manhattanDist(const QPoint& a, const QPoint& b) const
{
    return qAbs(a.x() - b.x()) + qAbs(a.y() - b.y());
}

QRect LayoutCanvas::normalizedRect(const QPoint& a, const QPoint& b) const
{
    const int x1 = qMin(a.x(), b.x());
    const int y1 = qMin(a.y(), b.y());
    const int x2 = qMax(a.x(), b.x());
    const int y2 = qMax(a.y(), b.y());
    return QRect(QPoint(x1, y1), QPoint(x2, y2));
}

void LayoutCanvas::mousePressEvent(QMouseEvent* e)
{
    // Common EDA behavior: left button initiates bbox gesture candidate
    m_pressed = true;
    m_dragging = false;
    m_pressPos = e->pos();
    m_origin = e->pos();
    m_pressButton = e->button();
    m_lastPreviewRect = QRect();

    e->accept();
}

void LayoutCanvas::mouseMoveEvent(QMouseEvent* e)
{
    if (!m_pressed) {
        e->ignore();
        return;
    }

    const QPoint cur = e->pos();

    if (!m_dragging) {
        if (manhattanDist(cur, m_pressPos) >= m_dragThreshold)
            m_dragging = true;
    }

    // Only treat left-button press as bbox gesture source
    if (m_dragging && m_pressButton == Qt::LeftButton) {
        const QRect r = normalizedRect(m_origin, cur);
        if (r != m_lastPreviewRect) {
            m_lastPreviewRect = r;
            emit bboxPreview(r);
        }
    }

    e->accept();
}

void LayoutCanvas::mouseReleaseEvent(QMouseEvent* e)
{
    if (!m_pressed) {
        e->ignore();
        return;
    }

    m_pressed = false;

    const QPoint cur = e->pos();
    const Qt::MouseButton releaseButton = e->button();

    // Only respond to the same button that started the gesture
    if (releaseButton != m_pressButton) {
        m_dragging = false;
        m_pressButton = Qt::NoButton;
        m_lastPreviewRect = QRect();
        e->accept();
        return;
    }

    if (m_dragging && m_pressButton == Qt::LeftButton) {
        const QRect r = normalizedRect(m_origin, cur);
        emit bboxCommitted(r);
    } else {
        emit clicked(cur, releaseButton, e->modifiers());
    }

    m_dragging = false;
    m_pressButton = Qt::NoButton;
    m_lastPreviewRect = QRect();

    e->accept();
}

void LayoutCanvas::paintEvent(QPaintEvent*)
{
    QPainter p(this);

    // White background
    p.fillRect(rect(), Qt::white);

    // Light gray border
    p.setPen(QPen(QColor(180, 180, 180), 1));
    p.drawRect(rect().adjusted(0, 0, -1, -1));
}
