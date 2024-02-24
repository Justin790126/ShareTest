#include <QtGui>

class CursorEventFilter : public QObject {
public:
    explicit CursorEventFilter(QObject *parent = NULL) : QObject(parent) {}

protected:
    bool eventFilter(QObject *obj, QEvent *event) override {
        if (obj->inherits("QListWidget")) {
            if (event->type() == QEvent::Enter) {
                // QPixmap cursorPixmap(":/icons/cursor.png");
                // QCursor cursor(cursorPixmap);
                // QApplication::setOverrideCursor(cursor);
                QApplication::setOverrideCursor(Qt::PointingHandCursor);
                return true;
            } else if (event->type() == QEvent::Leave) {
                QApplication::restoreOverrideCursor();
                return true;
            }
        }
        return false;
    }
};

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);

    QMainWindow mainWindow;
    mainWindow.setWindowTitle("List Widget with Cursor Change");

    // Create a central widget
    QWidget *centralWidget = new QWidget;
    mainWindow.setCentralWidget(centralWidget);

    // Create a layout for the central widget
    QVBoxLayout *layout = new QVBoxLayout(centralWidget);

    // Create a QListWidget
    QListWidget *listWidget = new QListWidget;
    listWidget->addItem("Item 1");
    listWidget->addItem("Item 2");
    listWidget->addItem("Item 3");

    // Install custom event filter on the QListWidget
    CursorEventFilter eventFilter;
    listWidget->installEventFilter(&eventFilter);

    // Add the QListWidget to the layout
    layout->addWidget(listWidget);

    // Show the main window
    mainWindow.show();

    return app.exec();
}
