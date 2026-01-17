#include "LayoutCanvas.h"

#include <QPainter>
#include <QtGlobal>

LayoutCanvas::LayoutCanvas(QWidget* parent)
    : QWidget(parent),
      m_pressed(false),
      m_dragging(false),
      m_dragThreshold(3),
      m_pressButton(Qt::NoButton),
      m_haveAnchor(false),
      m_anchorButton(Qt::NoButton),
      m_lastPreviewRect()
{
    setMouseTracking(true);
    setFocusPolicy(Qt::StrongFocus);
}

int LayoutCanvas::manhattanDist(const QPoint& a, const QPoint& b) const
{
    return qAbs(a.x() - b.x()) + qAbs(a.y() - b.y());
}

QRect LayoutCanvas::normalizedRect(const QPoint& a, const QPoint& b) const
{
    return QRect(QPoint(qMin(a.x(), b.x()), qMin(a.y(), b.y())),
                 QPoint(qMax(a.x(), b.x()), qMax(a.y(), b.y())));
}

void LayoutCanvas::mousePressEvent(QMouseEvent* e)
{
    m_pressed = true;
    m_dragging = false;
    m_pressPos = e->pos();
    m_pressButton = e->button();
    e->accept();
}

void LayoutCanvas::mouseMoveEvent(QMouseEvent* e)
{
    MouseState s;
    s.cursorPos = e->pos();

    // Hover preview after first click anchor (no press)
    if (!m_pressed) {
        if (m_haveAnchor) {
            m_lastPreviewRect = normalizedRect(m_anchor, e->pos());

            s.flow = MouseState::Flow_ClickClick;
            s.hasPreview = true;
            s.previewRect = m_lastPreviewRect;

            emit mouseUpdate(s);
            e->accept();
            return;
        }
        e->ignore();
        return;
    }

    // Drag detection while pressed
    if (!m_dragging &&
        manhattanDist(e->pos(), m_pressPos) >= m_dragThreshold)
        m_dragging = true;

    // Drag preview (Left only)
    if (m_dragging && m_pressButton == Qt::LeftButton) {
        const QPoint origin = m_haveAnchor ? m_anchor : m_pressPos;
        m_lastPreviewRect = normalizedRect(origin, e->pos());

        s.flow = MouseState::Flow_Drag;
        s.hasPreview = true;
        s.previewRect = m_lastPreviewRect;

        emit mouseUpdate(s);
    }

    e->accept();
}

void LayoutCanvas::mouseReleaseEvent(QMouseEvent* e)
{
    MouseState s;
    s.cursorPos = e->pos();
    s.button = e->button();

    // Middle release
    if (s.button == Qt::MiddleButton) {
        s.flow = MouseState::Flow_Middle;
        emit mouseRelease(s);
        m_pressed = false;
        return;
    }

    // If we were not in pressed state, treat as other
    if (!m_pressed) {
        s.flow = MouseState::Flow_Other;
        emit mouseRelease(s);
        return;
    }

    m_pressed = false;

    if (s.button != m_pressButton) {
        s.flow = MouseState::Flow_Other;
        emit mouseRelease(s);
        return;
    }

    // Drag commit (Left only)
    if (m_dragging && m_pressButton == Qt::LeftButton) {
        s.flow = MouseState::Flow_Drag;
        s.hasPreview = !m_lastPreviewRect.isNull();
        s.previewRect = m_lastPreviewRect;

        s.committed = true;
        s.committedRect = m_lastPreviewRect;

        // reset state
        m_dragging = false;
        m_haveAnchor = false;
        m_anchorButton = Qt::NoButton;
        m_lastPreviewRect = QRect();
        m_pressButton = Qt::NoButton;

        emit mouseRelease(s);
        return;
    }

    // Click-click (Left or Right)
    if (m_pressButton == Qt::LeftButton || m_pressButton == Qt::RightButton) {
        s.flow = MouseState::Flow_ClickClick;

        if (!m_haveAnchor) {
            // first click sets anchor
            m_anchor = e->pos();
            m_haveAnchor = true;
            m_anchorButton = m_pressButton;

            const QRect r = QRect(m_anchor, m_anchor);
            m_lastPreviewRect = r;

            s.hasPreview = true;
            s.previewRect = r;

            // not committed yet
            s.committed = false;

        } else if (m_pressButton == m_anchorButton) {
            // second click commits
            const QRect r = normalizedRect(m_anchor, e->pos());

            s.hasPreview = !m_lastPreviewRect.isNull();
            s.previewRect = m_lastPreviewRect;

            s.committed = true;
            s.committedRect = r;

            // reset anchor
            m_haveAnchor = false;
            m_anchorButton = Qt::NoButton;
            m_lastPreviewRect = QRect();

        } else {
            // mismatched button cancels
            s.flow = MouseState::Flow_Other;

            m_haveAnchor = false;
            m_anchorButton = Qt::NoButton;
            m_lastPreviewRect = QRect();
        }

        m_dragging = false;
        m_pressButton = Qt::NoButton;

        emit mouseRelease(s);
        return;
    }

    // Other
    s.flow = MouseState::Flow_Other;
    m_dragging = false;
    m_pressButton = Qt::NoButton;
    emit mouseRelease(s);
}

void LayoutCanvas::paintEvent(QPaintEvent*)
{
    // Canvas background only; preview is drawn by MainWindow rubber band overlay
    QPainter p(this);
    p.fillRect(rect(), Qt::white);
    p.setPen(QPen(QColor(180, 180, 180), 1));
    p.drawRect(rect().adjusted(0, 0, -1, -1));
}
