#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "LayoutCanvas.h"

class QRubberBand;
class LayoutCanvas;

class MainWindow : public QMainWindow
{
    Q_OBJECT
public:
    explicit MainWindow(QWidget* parent = 0);

private slots:
    void onMouseUpdate(const MouseState& s);
    void onMouseRelease(const MouseState& s);

private:
    void showBand(const QRect& r);
    void hideBand();

private:
    LayoutCanvas* m_canvas;
    QRubberBand*  m_band;
};

#endif
