#ifndef LAYOUTCANVAS_H
#define LAYOUTCANVAS_H

#include <QWidget>
#include <QPoint>
#include <QRect>

class LayoutCanvas : public QWidget
{
    Q_OBJECT
public:
    enum Mode {
        Mode_None = 0,
        Mode_Select = 1,
        Mode_Pan = 2,
        Mode_Simulation = 3
    };

    explicit LayoutCanvas(QWidget* parent = 0);

    int  mode() const { return m_iMode; }
    void setMode(int m);

    void setDragThreshold(int px) { m_dragThreshold = px; }
    int  dragThreshold() const { return m_dragThreshold; }

signals:
    void modeChanged(int newMode);

    // BBox gesture outputs (view coords)
    void bboxPreview(const QRect& viewRect);    // during dragging
    void bboxCommitted(const QRect& viewRect);  // on release if dragged

    void clicked(const QPoint& pos, Qt::MouseButton button, Qt::KeyboardModifiers mods);

protected:
    virtual void mousePressEvent(QMouseEvent* e);
    virtual void mouseMoveEvent(QMouseEvent* e);
    virtual void mouseReleaseEvent(QMouseEvent* e);
    virtual void paintEvent(QPaintEvent* e);

private:
    int   manhattanDist(const QPoint& a, const QPoint& b) const;
    QRect normalizedRect(const QPoint& a, const QPoint& b) const;

private:
    int m_iMode;

    // gesture flags/state (kept inside LayoutCanvas)
    bool m_pressed;
    bool m_dragging;
    int  m_dragThreshold;

    QPoint m_pressPos;
    QPoint m_origin;
    Qt::MouseButton m_pressButton;

    QRect m_lastPreviewRect; // avoid spamming same rect
};

#endif // LAYOUTCANVAS_H
