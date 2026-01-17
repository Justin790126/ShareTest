#ifndef LAYOUTCANVAS_H
#define LAYOUTCANVAS_H

#include <QWidget>
#include <QPoint>
#include <QRect>
#include <QMouseEvent>

struct MouseState
{
    enum Flow {
        Flow_None = 0,
        Flow_ClickClick,   // click / move / click (L or R)
        Flow_Drag,         // drag preview/release (Left)
        Flow_Middle,       // middle release
        Flow_Other
    };

    Flow flow;
    Qt::MouseButton button;

    // current cursor position (for debug/hud if needed)
    QPoint cursorPos;

    // preview info (valid on mouseUpdate, and may also be valid on mouseRelease)
    bool  hasPreview;
    QRect previewRect;

    // commit info (valid on mouseRelease if committed == true)
    bool  committed;
    QRect committedRect;

    MouseState()
        : flow(Flow_None),
          button(Qt::NoButton),
          cursorPos(),
          hasPreview(false),
          previewRect(),
          committed(false),
          committedRect()
    {}
};

class LayoutCanvas : public QWidget
{
    Q_OBJECT
public:
    explicit LayoutCanvas(QWidget* parent = 0);

    void setDragThreshold(int px) { m_dragThreshold = px; }

signals:
    void mouseUpdate(const MouseState& state);   // during move/preview
    void mouseRelease(const MouseState& state);  // on release

protected:
    virtual void mousePressEvent(QMouseEvent* e);
    virtual void mouseMoveEvent(QMouseEvent* e);
    virtual void mouseReleaseEvent(QMouseEvent* e);
    virtual void paintEvent(QPaintEvent* e);

private:
    int   manhattanDist(const QPoint& a, const QPoint& b) const;
    QRect normalizedRect(const QPoint& a, const QPoint& b) const;

private:
    bool m_pressed;
    bool m_dragging;
    int  m_dragThreshold;

    QPoint m_pressPos;
    Qt::MouseButton m_pressButton;

    bool   m_haveAnchor;
    QPoint m_anchor;
    Qt::MouseButton m_anchorButton;

    QRect m_lastPreviewRect;
};

#endif // LAYOUTCANVAS_H
