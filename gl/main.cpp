#include <QApplication>
#include <QGLWidget>
#include <QMessageBox>
#include <QMainWindow>

class GLTestWidget : public QGLWidget {
public:
    GLTestWidget(QWidget *parent = 0) : QGLWidget(parent) {}

protected:
    void initializeGL() {
        // If we reach here, OpenGL context is initialized
    }

    void paintGL() {
        // Minimal OpenGL rendering to verify functionality
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
    }
};

class MainWindow : public QMainWindow {
public:
    MainWindow() {
        GLTestWidget *glWidget = new GLTestWidget(this);
        setCentralWidget(glWidget);

        // Check if the widget is valid and OpenGL is supported
        if (glWidget->isValid()) {
            QMessageBox::information(this, "OpenGL Support",
                                   "OpenGL is supported on this system!");
        } else {
            QMessageBox::warning(this, "OpenGL Support",
                                "OpenGL is NOT supported on this system.");
        }
    }
};

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);

    // Check if OpenGL is available before creating the widget
    if (!QGLFormat::hasOpenGL()) {
        QMessageBox::critical(0, "OpenGL Support",
                             "This system does not support OpenGL.");
        return 1;
    }

    MainWindow window;
    window.resize(400, 300);
    window.show();

    return app.exec();
}