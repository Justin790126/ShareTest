#ifndef DRAWINGWIDGET_H
#define DRAWINGWIDGET_H

#include <QWidget>

class DrawingWidget : public QWidget {
    Q_OBJECT
public:
    DrawingWidget(QWidget *parent = nullptr);

protected:
    void paintEvent(QPaintEvent *event) override;
    void mousePressEvent(QMouseEvent *event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void mouseReleaseEvent(QMouseEvent *event) override;
    void wheelEvent(QWheelEvent *event) override;

private:
    qreal zoomFactor;
    QPoint panOffset;
    QPoint lastPos;
    bool drag;
};

#endif // DRAWINGWIDGET_H
