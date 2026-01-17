#include "MainWindow.h"
#include "LayoutCanvas.h"

#include <QRubberBand>
#include <QStatusBar>
#include <QDebug>

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent),
      m_canvas(0),
      m_band(0)
{
    m_canvas = new LayoutCanvas(this);
    setCentralWidget(m_canvas);

    // Rubber band overlay
    m_band = new QRubberBand(QRubberBand::Rectangle, m_canvas);
    m_band->setStyleSheet(
        "QRubberBand {"
        "  border: 1px solid rgb(80,80,80);"
        "  background-color: rgba(120,120,120,60);"
        "}"
    );
    m_band->hide();

    connect(m_canvas, SIGNAL(mouseUpdate(MouseState)),
            this, SLOT(onMouseUpdate(MouseState)));
    connect(m_canvas, SIGNAL(mouseRelease(MouseState)),
            this, SLOT(onMouseRelease(MouseState)));

    statusBar()->showMessage("Ready (Preview via mouseUpdate, Commit via mouseRelease)");
}

void MainWindow::showBand(const QRect& r)
{
    if (!m_band->isVisible())
        m_band->show();

    QRect rr = r;
    if (rr.width() == 0)  rr.setWidth(1);
    if (rr.height() == 0) rr.setHeight(1);

    m_band->setGeometry(rr);
}

void MainWindow::hideBand()
{
    if (m_band->isVisible())
        m_band->hide();
}

void MainWindow::onMouseUpdate(const MouseState& s)
{
    if (s.hasPreview) {
        showBand(s.previewRect);
    }
}

void MainWindow::onMouseRelease(const MouseState& s)
{
    if (s.flow == MouseState::Flow_Middle) {
        // typical: middle cancels preview
        hideBand();
        statusBar()->showMessage("Middle release");
        return;
    }

    // If committed, hide preview and handle ROI
    if (s.committed) {
        hideBand();
        const QRect& r = s.committedRect;

        const QString msg = QString("Committed ROI: x=%1 y=%2 w=%3 h=%4 (flow=%5 btn=%6)")
                                .arg(r.x()).arg(r.y())
                                .arg(r.width()).arg(r.height())
                                .arg((int)s.flow)
                                .arg((int)s.button);
        statusBar()->showMessage(msg);
        qDebug() << msg;
        return;
    }

    // For first click of click-click, we usually keep showing the anchor (0-size rect)
    if (s.hasPreview) {
        showBand(s.previewRect);
    } else {
        // Other releases: hide
        hideBand();
    }
}
