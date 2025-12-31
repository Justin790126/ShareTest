#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QRect>
#include <QPoint>

class LayoutCanvas;
class QRubberBand;

class MainWindow : public QMainWindow
{
    Q_OBJECT
public:
    explicit MainWindow(QWidget* parent = 0);

private slots:
    // From canvas
    void onBboxPreview(const QRect& viewRect);
    void onBboxCommitted(const QRect& viewRect);
    void onClicked(const QPoint& pos, Qt::MouseButton b, Qt::KeyboardModifiers mods);
    void onModeChanged(int newMode);

    // Demo toolbar actions
    void setModeSelect();
    void setModePan();
    void setModeSimulation();

private:
    bool shouldShowRubberBandByMode() const;
    void showBand(const QRect& r);
    void hideBand();

private:
    LayoutCanvas* m_canvas;
    QRubberBand*  m_band;
};

#endif // MAINWINDOW_H
