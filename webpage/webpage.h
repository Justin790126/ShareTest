#ifndef DRAWINGWIDGET_H
#define DRAWINGWIDGET_H

#include <QWidget>
#include <QMessageBox>
#include <QMenu>
#include <QAction>

class Webpage : public QWidget {
    Q_OBJECT
public:
    Webpage(QWidget *parent = nullptr);

protected:
    void paintEvent(QPaintEvent *event) override;
    void mousePressEvent(QMouseEvent *event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void mouseReleaseEvent(QMouseEvent *event) override;
    void wheelEvent(QWheelEvent *event) override;
    void contextMenuEvent(QContextMenuEvent *event) override;

private:
    qreal zoomFactor;
    QPoint panOffset;
    QPoint lastPos;
    bool drag;

private slots:
    void onAction1Triggered() {
        QMessageBox::information(this, "Action", "Action 1 triggered");
    }

    void onAction2Triggered() {
        QMessageBox::information(this, "Action", "Action 2 triggered");
    }

    void onAction3Triggered() {
        QMessageBox::information(this, "Action", "Action 3 triggered");
    }
};

#endif // DRAWINGWIDGET_H
