#include <QtGui/QApplication>
#include <QtGui/QWidget>
#include <QtGui/QPainter>
#include <QtGui/QMouseEvent>
#include <QtGui/QGraphicsScene>
#include <QtGui/QGraphicsView>
#include <QtGui/QGraphicsProxyWidget>
#include <QtGui/QVBoxLayout>

// Custom QWidget with overridden events
class CustomWidget : public QWidget {
public:
    CustomWidget(QWidget* parent = 0) : QWidget(parent), isRed(true) {
        setFixedSize(200, 100); // Fixed size for the widget
        setMouseTracking(true); // Enable mouse tracking if needed
    }

protected:
    void paintEvent(QPaintEvent* /*event*/) {
        QPainter painter(this);
        // Draw a colored rectangle
        painter.setBrush(isRed ? Qt::red : Qt::blue);
        painter.drawRect(rect().adjusted(0, 0, -1, -1));
        // Draw text
        painter.setPen(Qt::white);
        painter.drawText(rect(), Qt::AlignCenter, "Click to toggle color");
    }

    void mousePressEvent(QMouseEvent* event) {
        if (event->button() == Qt::LeftButton) {
            isRed = !isRed; // Toggle color
            update(); // Request repaint
        }
        QWidget::mousePressEvent(event); // Call base class handler
    }

private:
    bool isRed; // State for toggling color
};

int main(int argc, char* argv[]) {
    QApplication app(argc, argv);

    // Create the QGraphicsScene
    QGraphicsScene scene;
    scene.setSceneRect(0, 0, 400, 300); // Define the scene's coordinate system

    // Create the custom widget
    CustomWidget* customWidget = new CustomWidget;

    // Create a QGraphicsProxyWidget to embed the custom widget
    QGraphicsProxyWidget* proxyWidget = scene.addWidget(customWidget);
    proxyWidget->setPos(100, 100); // Position the widget in the scene
    proxyWidget->setZValue(1); // Optional: Set z-order

    // Optional: Apply transformations (e.g., rotation)
    proxyWidget->setRotation(10); // Rotate the widget by 10 degrees

    // Create the QGraphicsView to display the scene
    QGraphicsView view(&scene);
    view.setWindowTitle("Custom Widget in QGraphics Framework");
    view.resize(450, 350);
    view.setRenderHint(QPainter::Antialiasing); // Optional: Smooth rendering

    // Show the view
    view.show();

    return app.exec();
}