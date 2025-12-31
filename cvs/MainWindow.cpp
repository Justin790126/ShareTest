#include "MainWindow.h"
#include "LayoutCanvas.h"

#include <QRubberBand>
#include <QToolBar>
#include <QAction>
#include <QStatusBar>
#include <QDebug>

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent),
      m_canvas(0),
      m_band(0)
{
    m_canvas = new LayoutCanvas(this);
    setCentralWidget(m_canvas);

    // Rubber band owned/managed by MainWindow, but parented to canvas so it draws on top.
    m_band = new QRubberBand(QRubberBand::Rectangle, m_canvas);
    m_band->setStyleSheet(
        "QRubberBand {"
        "  border: 1px solid rgb(80,80,80);"
        "  background-color: rgba(120,120,120,60);"
        "}"
    );
    m_band->hide();

    // Connect canvas signals
    connect(m_canvas, SIGNAL(bboxPreview(QRect)), this, SLOT(onBboxPreview(QRect)));
    connect(m_canvas, SIGNAL(bboxCommitted(QRect)), this, SLOT(onBboxCommitted(QRect)));
    connect(m_canvas, SIGNAL(clicked(QPoint, Qt::MouseButton, Qt::KeyboardModifiers)),
            this, SLOT(onClicked(QPoint, Qt::MouseButton, Qt::KeyboardModifiers)));
    connect(m_canvas, SIGNAL(modeChanged(int)), this, SLOT(onModeChanged(int)));

    // Toolbar for mode switching (demo)
    QToolBar* tb = addToolBar("Tools");
    QAction* actSelect = tb->addAction("Select");
    QAction* actPan    = tb->addAction("Pan");
    QAction* actSim    = tb->addAction("Simulation");

    connect(actSelect, SIGNAL(triggered()), this, SLOT(setModeSelect()));
    connect(actPan,    SIGNAL(triggered()), this, SLOT(setModePan()));
    connect(actSim,    SIGNAL(triggered()), this, SLOT(setModeSimulation()));

    statusBar()->showMessage("Ready (Mode: Select)");

    m_canvas->setMode(LayoutCanvas::Mode_Select);
}

void MainWindow::setModeSelect()
{
    m_canvas->setMode(LayoutCanvas::Mode_Select);
    statusBar()->showMessage("Mode: Select (rubber-band ON)");
}

void MainWindow::setModePan()
{
    m_canvas->setMode(LayoutCanvas::Mode_Pan);
    statusBar()->showMessage("Mode: Pan (rubber-band OFF)");
}

void MainWindow::setModeSimulation()
{
    m_canvas->setMode(LayoutCanvas::Mode_Simulation);
    statusBar()->showMessage("Mode: Simulation (rubber-band ON)");
}

bool MainWindow::shouldShowRubberBandByMode() const
{
    const int m = m_canvas->mode();
    return (m == LayoutCanvas::Mode_Select) || (m == LayoutCanvas::Mode_Simulation);
}

void MainWindow::showBand(const QRect& r)
{
    if (!m_band->isVisible())
        m_band->show();
    m_band->setGeometry(r);
}

void MainWindow::hideBand()
{
    if (m_band->isVisible())
        m_band->hide();
}

void MainWindow::onModeChanged(int /*newMode*/)
{
    // When mode changes, immediately hide preview UI
    hideBand();
}

void MainWindow::onBboxPreview(const QRect& viewRect)
{
    if (!shouldShowRubberBandByMode()) {
        hideBand();
        return;
    }
    showBand(viewRect);
}

void MainWindow::onBboxCommitted(const QRect& viewRect)
{
    if (!shouldShowRubberBandByMode()) {
        hideBand();
        return;
    }

    hideBand();

    // Here is where you'd commit into state machine / dispatch action, per mode.
    // Example: in Simulation mode, treat rect as simulation ROI; in Select, as selection ROI.

    if (m_canvas->mode() == LayoutCanvas::Mode_Select) {
        const QString msg = QString("Select ROI committed: x=%1 y=%2 w=%3 h=%4")
                                .arg(viewRect.x()).arg(viewRect.y())
                                .arg(viewRect.width()).arg(viewRect.height());
        statusBar()->showMessage(msg);
        qDebug() << msg;
    } else if (m_canvas->mode() == LayoutCanvas::Mode_Simulation) {
        const QString msg = QString("Simulation ROI committed: x=%1 y=%2 w=%3 h=%4")
                                .arg(viewRect.x()).arg(viewRect.y())
                                .arg(viewRect.width()).arg(viewRect.height());
        statusBar()->showMessage(msg);
        qDebug() << msg;
    } else {
        // should not happen due to shouldShowRubberBandByMode()
    }
}

void MainWindow::onClicked(const QPoint& pos, Qt::MouseButton b, Qt::KeyboardModifiers mods)
{
    Q_UNUSED(mods);
    const QString msg = QString("Clicked (%1,%2) button=%3")
                            .arg(pos.x()).arg(pos.y()).arg((int)b);
    statusBar()->showMessage(msg);
    qDebug() << msg;
}
